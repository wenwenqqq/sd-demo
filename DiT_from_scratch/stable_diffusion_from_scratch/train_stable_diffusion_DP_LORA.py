import torch
import torch.nn as nn # DataParallel相关的导入和设备检测
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torchvision import transforms
from datasets import load_dataset
from stable_diffusion_model import load_vae_diffusion_model, StableDiffusion
from transformers import CLIPTokenizer, CLIPTextModel
from PIL import Image
import numpy as np
from tqdm.auto import tqdm
import os
import wandb
# 用封装的 peft 实现 lora 适配器的使用
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ===================== 新增：显存监控工具函数 =====================
def print_gpu_memory(epoch=None, mode="train"):
    if not torch.cuda.is_available():
        print("GPU不可用，跳过显存监控")
        return
    
    epoch_str = f"Epoch {epoch} " if epoch is not None else ""
    print(f"\n===== {epoch_str}{mode} 阶段 GPU 显存信息 =====")
    
    for gpu_id in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(gpu_id)
        total_mem = props.total_memory / 1024**3
        allocated_mem = torch.cuda.memory_allocated(gpu_id) / 1024**3
        reserved_mem = torch.cuda.memory_reserved(gpu_id) / 1024**3
        free_mem = total_mem - allocated_mem
        peak_allocated = torch.cuda.max_memory_allocated(gpu_id) / 1024**3
        
        print(f"GPU {gpu_id} ({props.name}):")
        print(f"  总显存: {total_mem:.2f} GiB")
        print(f"  已分配显存: {allocated_mem:.2f} GiB")
        print(f"  预留缓存显存: {reserved_mem:.2f} GiB")
        print(f"  可用显存: {free_mem:.2f} GiB")
        print(f"  峰值已分配显存: {peak_allocated:.2f} GiB")
    
    # 重置峰值统计
    for gpu_id in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(gpu_id)
    print("="*50 + "\n")

def print_model_modules(model, prefix=""):
    """递归打印模型所有模块名称和类型，定位注意力模块"""
    for name, module in model.named_modules():
        full_name = f"{prefix}.{name}" if prefix else name
        print(f"模块名称: {full_name:<50} 模块类型: {type(module).__name__}")

def find_attention_modules(model):
    """查找所有注意力相关模块，返回名称列表"""
    attn_modules = []
    for name, module in model.named_modules():
        # 匹配注意力相关关键词（根据实际模型调整）
        if any(key in name.lower() for key in ["attn", "attention", "qkv"]):
            attn_modules.append(name)
    return attn_modules

# 超参数
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    multi_gpu = True
    device_ids = None  # 使用所有可用的GPU
    print(f"多卡-{torch.cuda.device_count()}卡-DP可用")
else:
    multi_gpu = False
    device_ids = None
    print("单卡")
image_size = 512
latent_size = 64 # 潜在表示的宽和高，用于生成图像
n_epochs = 5
batch_size = 16 # 总batchsize
lr = 1e-4
num_timesteps = 1000
save_checkpoint_interval = 5
lambda_cons = 0.1  # 一致性损失的权重
max_lambda_cons = 1.0  # 最大一致性损失权重
epochs_to_max_lambda = n_epochs  # 达到最大权重所需的epoch数

# WandB 初始化
run = wandb.init(
    project="stable_diffusion_from_scratch",
    config={
        "batch_size": batch_size,
        "learning_rate": lr,
        "epochs": n_epochs,
        "num_timesteps": num_timesteps,
    },
)

# class AugmentedLatentDataset(Dataset):
#     def __init__(self, original_dataset, model, device, num_augmentations=5):
#         self.original_dataset = original_dataset
#         self.model = model
#         self.device = device
#         self.num_augmentations = num_augmentations

#         self.augment = transforms.Compose([
#             transforms.RandomHorizontalFlip(),
#             transforms.RandomRotation(10),
#             transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
#         ])

#     def __len__(self):
#         return len(self.original_dataset) * self.num_augmentations

#     def __getitem__(self, idx):
#         original_idx = idx // self.num_augmentations
#         original_item = self.original_dataset[original_idx]

#         image = original_item["images"]
#         text = original_item["text"]

#         # Apply augmentation
#         augmented_image = self.augment(image)

#         # Encode to latent space
#         # if multi_gpu:
#         #     model_encode = model.module.encode
#         # else:
#         #     model_encode = model.encode
#         with torch.no_grad():
#             latent = self.model.module.encode(augmented_image.unsqueeze(0).to(self.device))

#         return {"latents": latent.squeeze(0).cpu(), "text": text}

class AugmentedLatentDataset(Dataset):
    def __init__(self, original_dataset, model, device, num_augmentations=5):
        self.original_dataset = original_dataset
        self.model = model
        self.device = device
        self.num_augmentations = num_augmentations

        self.augment = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ])

    def __len__(self):
        return len(self.original_dataset) * self.num_augmentations

    def __getitem__(self, idx):
        original_idx = idx // self.num_augmentations
        original_item = self.original_dataset[original_idx]

        image = original_item["images"]
        text = original_item["text"]

        # Apply augmentation
        augmented_image = self.augment(image)

        # Encode to latent space（修复model.module的判断逻辑）
        with torch.no_grad():
            # 区分多卡/单卡的encode方法
            if hasattr(self.model, 'module'):
                latent = self.model.module.encode(augmented_image.unsqueeze(0).to(self.device))
            else:
                latent = self.model.encode(augmented_image.unsqueeze(0).to(self.device))

        return {"latents": latent.squeeze(0).cpu(), "text": text}

# 加载适合LoRA微调的数据集
# dataset = load_dataset("lambdalabs/pokemon-blip-captions", split="train")
dataset = load_dataset("diffusers/pokemon-gpt4-captions", split="train")

preprocess = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def transform(examples):
    images = [preprocess(image.convert("RGB")) for image in examples["image"]]
    return {"images": images, "text": examples["text"]}

dataset.set_transform(transform)

# 初始化合并的VAE+Diffusion模型
model = StableDiffusion(in_channels=3, latent_dim=4, image_size=512, diffusion_timesteps=1000, device=device)
checkpoint = torch.load('/data/bdd100k/sd/stable_diffusion_model_final.pth', map_location=device)
# model.load_state_dict(checkpoint['model_state_dict'])
# 多卡DP模型加载：移除module.前缀（如果有）
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in checkpoint.items():  # 注意：这里是checkpoint而非checkpoint['model_state_dict']
    # 移除module.前缀（多卡训练的模型权重会带这个前缀）
    name = k.replace('module.', '') if 'module.' in k else k
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)
print("sd预训练权重load完成")
model.load_vae('/data/bdd100k/sd/DiT_from_scratch/stable_diffusion_from_scratch/vae_model.pth')
print("VAE预训练权重load完成")
# model = load_vae_diffusion_model('vae_model.pth',
#                                  in_channels=3,
#                                  latent_dim=4,
#                                  image_size=512,
#                                  diffusion_timesteps=1000,
#                                  device=device)

if multi_gpu:
    model = nn.DataParallel(model, device_ids=device_ids) # 使用torch.nn.DataParallel包装模型
model.to(device)

# # ===================== 新增：打印UNet模块 =====================
# # 提取UNet模型（区分多卡/单卡）
# if multi_gpu:
#     unet = model.module.unet
# else:
#     unet = model.unet

# # 打印UNet所有模块（关键！找到正确的注意力模块名）
# print("\n========== UNet模型所有模块列表 ==========")
# print_model_modules(unet)

# # 查找注意力相关模块
# print("\n========== 注意力相关模块列表 ==========")
# attn_modules = find_attention_modules(unet)
# for idx, name in enumerate(attn_modules):
#     print(f"{idx+1}. {name}")
# print("=========================================\n")

# 配置LoRA
lora_config = LoraConfig(
    r=16,  # LoRA注意力维度
    lora_alpha=32,  # LoRA缩放参数
    target_modules=["query", "key", "value", "out_proj"],  # 只选择Attention内的Linear层
    lora_dropout=0.05,
    bias="none",
    task_type="CUSTOM",  # 核心修改：使用通用类型
)

# 为UNet添加LoRA适配器
if multi_gpu:
    model.module.unet = get_peft_model(model.module.unet, lora_config)
else:
    model.unet = get_peft_model(model.unet, lora_config)

# 打印可训练参数数量
if multi_gpu:
    model.module.unet.print_trainable_parameters()
else:
    model.unet.print_trainable_parameters()

# Create augmented datasets
train_dataset = AugmentedLatentDataset(dataset.select(range(0, 600)), model, device, num_augmentations=5)
val_dataset = dataset.select(range(600, 800))

# Create data loaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

# 加载 CLIP 模型
clip_local_path = "/data/bdd100k/sd/models--openai--clip-vit-base-patch32/snapshots/clip-vit-base-patch32"
tokenizer = CLIPTokenizer.from_pretrained(clip_local_path)
text_encoder = CLIPTextModel.from_pretrained(clip_local_path).to(device)

# WandB 监控
wandb.watch(model, log_freq=100)

# DP访问需要加.module
if multi_gpu:
    vae_module = model.module.vae
    unet_module = model.module.unet
else:
    vae_module = model.vae
    unet_module = model.unet

# 冻结VAE参数
for param in vae_module.parameters():
    param.requires_grad = False
# 确保UNet (diffusion model) 参数可训练
for param in unet_module.parameters():
    param.requires_grad = True

# 只优化LoRA参数
if multi_gpu:
    lora_parameters = model.module.unet.parameters()
else:
    lora_parameters = model.unet.parameters()

# optimizer = AdamW(lora_parameters, lr=lr, weight_decay=1e-4)
# scheduler = OneCycleLR(optimizer, max_lr=1e-4, epochs=n_epochs, steps_per_epoch=len(train_dataloader))
# 替换优化器/调度器配置
optimizer = AdamW(lora_parameters, lr=lr, weight_decay=1e-4, eps=1e-8)  # 增加eps避免除0
# 改用余弦退火调度器，更稳定
scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)  # 最低学习率1e-6，避免降到0

# 创建保存生成测试图像的目录
os.makedirs('stable_diffusion_results', exist_ok=True)

# 辅助损失函数：多样性损失
def diversity_loss(latents, use_cosine=False):
    """
    计算多样性损失，可选使用余弦相似度
    """
    batch_size = latents.size(0)
    latents_flat = latents.view(batch_size, -1)

    if use_cosine:
        # 使用余弦相似度
        latents_norm = F.normalize(latents_flat, p=2, dim=1)
        similarity = torch.mm(latents_norm, latents_norm.t())
    else:
        # 使用原始的点积相似度
        similarity = torch.mm(latents_flat, latents_flat.t())

    # 移除对角线上的自相似度
    similarity = similarity - torch.eye(batch_size, device=latents.device)

    return similarity.sum() / (batch_size * (batch_size - 1))

diversity_weight = 0.01  # 多样性损失起始权重

# 训练循环
for epoch in range(n_epochs):
    model.train()
    # print_gpu_memory(epoch=epoch, mode="训练前")
    progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}/{n_epochs}")
    epoch_loss = 0.0
    num_batches = 0

    # 更新一致性损失权重
    current_lambda_cons = min(lambda_cons * (epoch + 1) / epochs_to_max_lambda, max_lambda_cons)

    # 训练模型
    for batch in train_dataloader:
        latents = batch["latents"].to(device)
        text = batch["text"]

        # 添加噪声
        timesteps = torch.randint(0, num_timesteps, (latents.shape[0],), device=device).long()
        noisy_latents, noise = model.module.noise_scheduler.add_noise(latents, timesteps)

        # 使用 CLIP 模型编码文本
        text_inputs = tokenizer(text, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
        text_embeddings = text_encoder(text_inputs.input_ids.to(device)).last_hidden_state

        # 预测噪声
        noise_pred = model(noisy_latents, timesteps, text_embeddings)
        mse_loss = F.mse_loss(noise_pred, noise)
        div_loss = diversity_loss(noisy_latents, use_cosine=True)

        # 计算去噪后的潜在表示
        alpha_t = model.module.noise_scheduler.alphas[timesteps][:, None, None, None]
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
        predicted_latents = (noisy_latents - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t
        cons_loss = F.mse_loss(predicted_latents, latents)


        # 组合损失
        total_loss = mse_loss + diversity_weight * div_loss + cons_loss * current_lambda_cons
        epoch_loss += total_loss.item()
        num_batches += 1

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()  # OneCycleLR 学习率调度器

        # 动态调整多样性损失的权重
        if epoch % 10 == 0:
            diversity_weight = min(diversity_weight * 1.05, 0.1)  # 逐渐增加权重，但设置上限

        progress_bar.update(1)
        # print_gpu_memory(epoch=epoch, mode="训练后")
        progress_bar.set_postfix({"loss": epoch_loss / num_batches})

    average_train_loss = epoch_loss / num_batches

    # 验证集上评估模型
    model.eval()
    # print_gpu_memory(epoch=epoch, mode="验证前")
    val_loss = 0.0
    val_batches = 0
    with torch.no_grad():
        for batch in val_dataloader:
            data = batch["images"].to(device)
            latents = model.module.encode(data)
            text = batch["text"]

            # 添加噪声
            timesteps = torch.randint(0, num_timesteps, (latents.shape[0],), device=device).long()
            noisy_latents, noise = model.module.noise_scheduler.add_noise(latents, timesteps)

            # 使用 CLIP 模型编码文本
            text_inputs = tokenizer(text, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
            text_embeddings = text_encoder(text_inputs.input_ids.to(device)).last_hidden_state

            # 预测噪声
            noise_pred = model(noisy_latents, timesteps, text_embeddings)
            mse_loss = F.mse_loss(noise_pred, noise)

            val_loss += mse_loss.item()
            val_batches += 1
    # print_gpu_memory(epoch=epoch, mode="验证后")
    average_val_loss = val_loss / val_batches

    # scheduler.step()

    wandb.log({
        "epoch": epoch,
        "learning_rate": scheduler.get_last_lr()[0],
        "train_loss": average_train_loss,
        "val_loss": average_val_loss,
    })

    # # 保存模型检查点
    # if (epoch + 1) % save_checkpoint_interval == 0:
    #     torch.save({
    #         'epoch': epoch,
    #         # 保存LoRA权重
    #         'model_state_dict': model.module.unet.state_dict() if multi_gpu else model.unet.state_dict(),
    #         'optimizer_state_dict': optimizer.state_dict(),
    #         'scheduler_state_dict': scheduler.state_dict(),
    #         'train_loss': epoch_loss,
    #         'val_loss': val_loss,
    #     }, f'stable_diffusion_results/stable_diffusion_lora_checkpoint_epoch_{epoch+1}.pth')

    # 替换原有保存检查点的代码
    if (epoch + 1) % save_checkpoint_interval == 0:
        # 1. 保存纯LoRA适配器权重（PEFT标准格式，仅几MB）
        lora_save_path = f'/data/bdd100k/sd/stable_diffusion_results/lora_weights_epoch_{epoch+1}'
        if multi_gpu:
            model.module.unet.save_pretrained(lora_save_path)
        else:
            model.unet.save_pretrained(lora_save_path)
        
        # 2. 可选：保存优化器/调度器状态（用于断点续训）
        torch.save({
            'epoch': epoch,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': epoch_loss,
            'val_loss': val_loss,
        }, f'/data/bdd100k/sd/stable_diffusion_results/training_state_epoch_{epoch+1}.pth')
        
        print(f"LoRA权重已保存到: {lora_save_path}")
        print(f"训练状态已保存到: /data/bdd100k/sd/stable_diffusion_results/training_state_epoch_{epoch+1}.pth")

    # 生成测试图像
    if (epoch + 1) % save_checkpoint_interval == 0:
        model.eval()
        with torch.no_grad():
            sample_text = ["a water type pokemon", "a red pokemon with a red fire tail"]
            text_input = tokenizer(sample_text, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
            text_embeddings = text_encoder(text_input.input_ids.to(device)).last_hidden_state

            # 使用模型的sample方法生成图像
            sampled_latents = model.module.sample(text_embeddings, latent_size=latent_size, batch_size=len(sample_text), guidance_scale=7.5, device=device)

            # 使用VAE解码器将潜在表示解码回像素空间
            sampled_images = model.module.decode(sampled_latents)

            # 保存生成的图像
            for i, img in enumerate(sampled_images):
                img = img * 0.5 + 0.5  # Rescale to [0, 1]
                img = img.detach().cpu().permute(1, 2, 0).numpy()
                img = (img * 255).astype(np.uint8)
                img_pil = Image.fromarray(img)
                img_pil.save(f'/data/bdd100k/sd/stable_diffusion_results/generated_image_epoch_{epoch+1}_sample_{i}.png')

            wandb.log({f"generated_image_{i}": wandb.Image(sampled_images[i]) for i in range(len(sample_text))})

# torch.save(model.state_dict(), 'stable_diffusion_model_final.pth')
torch.save(model.module.state_dict() if multi_gpu else model.state_dict(), 'stable_diffusion_model_final.pth') # DP适配
wandb.finish()