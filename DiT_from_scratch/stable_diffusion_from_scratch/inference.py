import torch
import os
from collections import OrderedDict
from diffusion_model import UNet_Transformer, NoiseScheduler, sample_cfg, sample
from stable_diffusion_model import StableDiffusion
from transformers import CLIPTokenizer, CLIPTextModel
from peft import PeftModel, LoraConfig
from PIL import Image
import numpy as np

# ===================== 超参数 =====================
device = "cuda" if torch.cuda.is_available() else "cpu"
image_size = 64          # latent_size
vae_image_size = 512     # VAE 输入/输出像素尺寸
in_channels = 4          # UNet 输入 latent 通道数

# ✅ 修复 1：减少推理时间步（1000→100，速度提升 10 倍）
num_timesteps = 1000     # 训练时的时间步（用于 noise_scheduler）
inference_steps = 100    # ✅ 推理时的实际步数（可调整 50-200）

# ===================== LoRA 配置 =====================
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["query", "key", "value", "out_proj"],
    bias="none",
    task_type="CUSTOM",
)

# ===================== 路径配置 =====================
sd_model_path = "/data/bdd100k/sd/stable_diffusion_model_final.pth"
lora_ckpt_dir = "/data/bdd100k/sd/stable_diffusion_results/lora_weights_epoch_5"
clip_local_path = "/data/bdd100k/sd/models--openai--clip-vit-base-patch32/snapshots/clip-vit-base-patch32/"

# ===================== 初始化模型 =====================
print("🔄 初始化 StableDiffusion 模型...")
model = StableDiffusion(
    in_channels=3,
    latent_dim=4,
    image_size=vae_image_size,
    diffusion_timesteps=num_timesteps,
    device=device
)

# ===================== 加载权重 =====================
print(f"🔄 加载 SD 模型权重：{sd_model_path}")
checkpoint = torch.load(sd_model_path, map_location=device, weights_only=True)
state_dict = checkpoint.get('model_state_dict', checkpoint)

unet_state_dict = OrderedDict()
for k, v in state_dict.items():
    k_clean = k.replace('module.', '') if k.startswith('module.') else k
    if k_clean.startswith('unet.'):
        unet_state_dict[k_clean.replace('unet.', '')] = v

model.unet.load_state_dict(unet_state_dict, strict=False)
print("✅ UNet 基础权重加载完成")

vae_state_dict = {k.replace('module.', '').replace('vae.', ''): v 
                  for k, v in state_dict.items() 
                  if k.replace('module.', '').startswith('vae.')}
if vae_state_dict:
    model.vae.load_state_dict(vae_state_dict)
    print("✅ VAE 权重加载完成")

model.to(device)
model.eval()

# ===================== 加载 LoRA =====================
print(f"🔄 加载 LoRA 权重：{lora_ckpt_dir}")
model.unet = PeftModel.from_pretrained(
    model.unet, 
    lora_ckpt_dir,
    config=lora_config,
    is_trainable=False
)
print("✅ LoRA 权重加载完成")

# ===================== 加载 CLIP =====================
print("🔄 加载 CLIP 文本编码器...")
tokenizer = CLIPTokenizer.from_pretrained(clip_local_path)
text_encoder = CLIPTextModel.from_pretrained(clip_local_path).to(device)
text_encoder.eval()
print("✅ CLIP 加载完成")

# ===================== 推理配置 =====================
os.makedirs("inference_output", exist_ok=True)

prompts = [
    "a green bird with a red tail and a black nose",
    "a cute pokemon with blue fur and yellow cheeks",
]
guidance_scale = 7.5

# ===================== 文本编码 =====================
print("🔄 编码文本条件...")
all_embeddings = []
for prompt in prompts:
    text_input = tokenizer(
        [prompt], 
        padding="max_length", 
        max_length=tokenizer.model_max_length, 
        truncation=True, 
        return_tensors="pt"
    )
    with torch.no_grad():
        embedding = text_encoder(text_input.input_ids.to(device)).last_hidden_state
    all_embeddings.append(embedding)
print("✅ 文本编码完成")

# ===================== 图像生成（CFG 采样） =====================
print(f"🎨 开始生成图像（CFG 采样，{inference_steps}步）...")

with torch.no_grad():
    for i, text_emb in enumerate(all_embeddings):
        text_emb = text_emb.to(device)
        
        latent = sample_cfg(
            model.unet,
            model.noise_scheduler,
            n_samples=1,
            in_channels=4,
            text_embeddings=text_emb,
            image_size=image_size,
            guidance_scale=guidance_scale
        )
        
        # VAE 解码
        image = model.vae.decode(latent)
        
        # ✅ 修复 2：先去除 batch 维度再 permute
        image = image.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
        image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        
        save_path = f"inference_output/generated_cfg_{i}.png"
        Image.fromarray(image).save(save_path)
        print(f"  ✓ 已保存：{save_path}")

# ===================== 普通采样（加速版） =====================
print(f"🎨 开始生成图像（普通采样，{inference_steps}步）...")

with torch.no_grad():
    for i, text_emb in enumerate(all_embeddings):
        text_emb = text_emb.to(device)
        x_t = torch.randn(1, 4, image_size, image_size).to(device)
        
        # 跳步采样加速
        skip = max(1, num_timesteps // inference_steps)
        timesteps = list(reversed(range(0, num_timesteps, skip)))[:inference_steps]
        
        for t in timesteps:
            # ✅ 关键修复：用 [t] 创建 1-d tensor，而不是 t
            t_tensor = torch.tensor([t], device=device, dtype=torch.long)  # ← 注意方括号
            
            x_t = sample(
                model.unet, 
                x_t, 
                model.noise_scheduler, 
                t_tensor,  # ← 传 Tensor 而非 int
                text_emb
            )
        
        # VAE 解码
        image = model.vae.decode(x_t)
        
        # 去除 batch 维度
        image = image.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
        image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        
        save_path = f"inference_output/generated_normal_{i}.png"
        Image.fromarray(image).save(save_path)
        print(f"  ✓ 已保存：{save_path}")