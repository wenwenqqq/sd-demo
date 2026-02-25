# precompute_latents.py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from stable_diffusion_model import StableDiffusion
from tqdm import tqdm
import os

# =============== 配置 ===============
image_size = 512
vae_path = "/data/bdd100k/sd/DiT_from_scratch/stable_diffusion_from_scratch/vae_model.pth"
output_dir = "./precomputed_latents"
os.makedirs(output_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# =============== 数据预处理 ===============
preprocess = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def transform(examples):
    images = [preprocess(img.convert("RGB")) for img in examples["image"]]
    return {"images": images, "text": examples["en_text"]}

# =============== 加载模型 ===============
print("Loading VAE model...")
model = StableDiffusion(
    in_channels=3,
    latent_dim=4,
    image_size=image_size,
    diffusion_timesteps=1000,
    device=device
)
model.load_vae(vae_path)
model.eval()
model.to(device)

# =============== 预计算函数 ===============
def precompute_and_save(split_name, indices, batch_size=8):
    """
    split_name: "train" or "val"
    indices: 要处理的数据索引范围，如 range(0, 600)
    """
    print(f"\nPrecomputing {split_name} latents...")
    
    # 加载原始数据集
    dataset = load_dataset("svjack/pokemon-blip-captions-en-zh", split="train")
    subset = dataset.select(indices)
    subset.set_transform(transform)
    
    dataloader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    all_latents = []
    all_texts = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Encoding {split_name}"):
            images = batch["images"].to(device)
            # Encode to latent space (mean of VAE encoder)
            latents = model.encode(images)  # shape: [B, 4, 64, 64]
            all_latents.append(latents.cpu())
            all_texts.extend(batch["text"])
    
    # 保存
    output_path = os.path.join(output_dir, f"pokemon_{split_name}_latents.pt")
    torch.save({
        "latents": torch.cat(all_latents, dim=0),  # [N, 4, 64, 64]
        "texts": all_texts,                        # List[str], length=N
        "indices": list(indices)
    }, output_path)
    print(f"Saved {split_name} latents to {output_path} | Total samples: {len(all_texts)}")

# =============== 执行预计算 ===============
if __name__ == "__main__":
    # 训练集: 0-599
    precompute_and_save("train", range(0, 600), batch_size=8)
    # 验证集: 600-799
    precompute_and_save("val", range(600, 800), batch_size=8)
    print("\n✅ Precomputation completed!")