import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import OneCycleLR
from torchvision import transforms
from datasets import load_dataset
from stable_diffusion_model import load_vae_diffusion_model, StableDiffusion
from transformers import CLIPTokenizer, CLIPTextModel
from PIL import Image
import numpy as np
from tqdm.auto import tqdm
import os
import wandb
from latent_dataset import PrecomputedLatentDataset
print(f"RANK={os.environ.get('RANK')}, LOCAL_RANK={os.environ.get('LOCAL_RANK')}, WORLD_SIZE={os.environ.get('WORLD_SIZE')}")

# ===================== 显存监控工具函数 =====================
def print_gpu_memory(rank, epoch=None, mode="train"):
    if not torch.cuda.is_available():
        return
    epoch_str = f"Epoch {epoch} " if epoch is not None else ""
    if rank == 0:
        print(f"\n===== {epoch_str}{mode} 阶段 GPU 显存信息 (rank={rank}) =====")
    for gpu_id in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(gpu_id)
        total_mem = props.total_memory / 1024**3
        allocated_mem = torch.cuda.memory_allocated(gpu_id) / 1024**3
        reserved_mem = torch.cuda.memory_reserved(gpu_id) / 1024**3
        free_mem = total_mem - allocated_mem
        peak_allocated = torch.cuda.max_memory_allocated(gpu_id) / 1024**3
        if rank == 0:
            print(f"GPU {gpu_id} ({props.name}):")
            print(f"  总显存: {total_mem:.2f} GiB")
            print(f"  已分配显存: {allocated_mem:.2f} GiB")
            print(f"  预留缓存显存: {reserved_mem:.2f} GiB")
            print(f"  可用显存: {free_mem:.2f} GiB")
            print(f"  峰值已分配显存: {peak_allocated:.2f} GiB")
    for gpu_id in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(gpu_id)

# 超参数（注意：batch_size 现为 per-GPU）
image_size = 512
latent_size = 64
n_epochs = 2
batch_size = 4  # 每卡 batch size（全局 = batch_size * world_size）
lr = 1e-4
num_timesteps = 1000
save_checkpoint_interval = 50
lambda_cons = 0.1
max_lambda_cons = 1.0
epochs_to_max_lambda = n_epochs
diversity_weight = 0.01

class AugmentedLatentDataset(Dataset):
    def __init__(self, original_dataset, model, device, num_augmentations=5):
        self.original_dataset = original_dataset
        self.model = model  # DDP 模型（带 .module）
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
        augmented_image = self.augment(image)
        with torch.no_grad():
            latent = self.model.module.encode(augmented_image.unsqueeze(0).to(self.device))
        return {"latents": latent.squeeze(0).cpu(), "text": text}

def diversity_loss(latents, use_cosine=False):
    batch_size = latents.size(0)
    latents_flat = latents.view(batch_size, -1)
    if use_cosine:
        latents_norm = F.normalize(latents_flat, p=2, dim=1)
        similarity = torch.mm(latents_norm, latents_norm.t())
    else:
        similarity = torch.mm(latents_flat, latents_flat.t())
    similarity = similarity - torch.eye(batch_size, device=latents.device)
    return similarity.sum() / (batch_size * (batch_size - 1))

def main():
    # ========== DDP 初始化 ==========
    assert torch.cuda.is_available(), "DDP requires CUDA"
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)
    # ===============================

    # ========== WandB（仅 rank=0）==========
    if rank == 0:
        run = wandb.init(
            project="stable_diffusion_from_scratch",
            config={
                "batch_size_per_gpu": batch_size,
                "global_batch_size": batch_size * world_size,
                "learning_rate": lr,
                "epochs": n_epochs,
                "num_timesteps": num_timesteps,
            },
        )
        os.makedirs('stable_diffusion_results', exist_ok=True)
    # ====================================

    # ========== 数据预处理 ==========
    preprocess = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    def transform(examples):
        images = [preprocess(img.convert("RGB")) for img in examples["image"]]
        return {"images": images, "text": examples["en_text"]}
    
    dataset = load_dataset("svjack/pokemon-blip-captions-en-zh", split="train")
    dataset.set_transform(transform)
    # ====================================

    # ========== 模型初始化 ==========
    model = StableDiffusion(
        in_channels=3,
        latent_dim=4,
        image_size=512,
        diffusion_timesteps=1000,
        device=device
    )
    model.load_vae('/data/bdd100k/sd/DiT_from_scratch/stable_diffusion_from_scratch/vae_model.pth')
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    # ====================================

    # ========== 数据集 & DataLoader ==========
    # train_full = dataset.select(range(0, 600))
    # val_full = dataset.select(range(600, 800))
    # # 注意：每个进程独立创建 AugmentedLatentDataset（避免共享模型状态）
    # train_dataset = AugmentedLatentDataset(train_full, model, device, num_augmentations=5)
    # val_dataset = val_full  # 验证集不增强
    train_dataset = PrecomputedLatentDataset(
    "./precomputed_latents/pokemon_train_latents.pt",
    augment_latent=True
    )
    val_dataset = PrecomputedLatentDataset(
        "./precomputed_latents/pokemon_val_latents.pt",
        augment_latent=False
    )

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    # ====================================

    # ========== CLIP 文本编码器 ==========
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    # ====================================

    # ========== 冻结 VAE，训练 UNet ==========
    for param in model.module.vae.parameters():
        param.requires_grad = False
    for param in model.module.unet.parameters():
        param.requires_grad = True
    # ====================================

    # ========== 优化器 & 调度器 ==========
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=1e-4
    )
    scheduler = OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=n_epochs,
        steps_per_epoch=len(train_dataloader)
    )
    # ====================================

    diversity_weight = 0.01

    # ========== 训练循环 ==========
    for epoch in range(n_epochs):
        train_sampler.set_epoch(epoch)  # 关键：每 epoch 重新打乱
        model.train()

        epoch_loss = 0.0
        num_batches = 0
        current_lambda_cons = min(lambda_cons * (epoch + 1) / epochs_to_max_lambda, max_lambda_cons)

        progress_bar = tqdm(total=len(train_dataloader), desc=f"Rank {rank} Epoch {epoch+1}/{n_epochs}", disable=(rank != 0))
        for batch in train_dataloader:
            latents = batch["latents"].to(device)
            text = batch["text"]

            timesteps = torch.randint(0, num_timesteps, (latents.shape[0],), device=device).long()
            noisy_latents, noise = model.module.noise_scheduler.add_noise(latents, timesteps)

            text_inputs = tokenizer(text, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
            text_embeddings = text_encoder(text_inputs.input_ids.to(device)).last_hidden_state

            noise_pred = model(noisy_latents, timesteps, text_embeddings)
            mse_loss = F.mse_loss(noise_pred, noise)
            div_loss = diversity_loss(noisy_latents, use_cosine=True)

            alpha_t = model.module.noise_scheduler.alphas[timesteps][:, None, None, None]
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
            predicted_latents = (noisy_latents - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t
            cons_loss = F.mse_loss(predicted_latents, latents)

            total_loss = mse_loss + diversity_weight * div_loss + cons_loss * current_lambda_cons
            epoch_loss += total_loss.item()
            num_batches += 1

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            # 动态调整多样性权重
            if epoch % 10 == 0:
                diversity_weight = min(diversity_weight * 1.05, 0.1)

            progress_bar.update(1)
            if rank == 0:
                # print_gpu_memory(rank, epoch=epoch, mode="batch间")
                progress_bar.set_postfix({"loss": epoch_loss / num_batches})

        average_train_loss = epoch_loss / num_batches

        # ========== 验证 ==========
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_dataloader:
                # data = batch["images"].to(device)
                # latents = model.module.encode(data)
                # text = batch["text"]
                latents = batch["latents"].to(device)  # 直接从 DataLoader 获取
                text = batch["text"]

                timesteps = torch.randint(0, num_timesteps, (latents.shape[0],), device=device).long()
                noisy_latents, noise = model.module.noise_scheduler.add_noise(latents, timesteps)

                text_inputs = tokenizer(text, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
                text_embeddings = text_encoder(text_inputs.input_ids.to(device)).last_hidden_state

                noise_pred = model(noisy_latents, timesteps, text_embeddings)
                mse_loss = F.mse_loss(noise_pred, noise)
                val_loss += mse_loss.item()
                val_batches += 1
        average_val_loss = val_loss / val_batches

        # ========== 日志 & 保存（仅 rank=0）==========
        if rank == 0:
            wandb.log({
                "epoch": epoch,
                "learning_rate": scheduler.get_last_lr()[0],
                "train_loss": average_train_loss,
                "val_loss": average_val_loss,
            })

            if (epoch + 1) % save_checkpoint_interval == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }, f'stable_diffusion_results/stable_diffusion_model_checkpoint_epoch_{epoch+1}.pth')

                # 生成测试图像
                model.eval()
                with torch.no_grad():
                    sample_text = ["a water type pokemon", "a red pokemon with a red fire tail"]
                    text_input = tokenizer(sample_text, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
                    text_embeddings = text_encoder(text_input.input_ids.to(device)).last_hidden_state

                    sampled_latents = model.module.sample(
                        text_embeddings,
                        latent_size=latent_size,
                        batch_size=len(sample_text),
                        guidance_scale=7.5,
                        device=device
                    )
                    sampled_images = model.module.decode(sampled_latents)

                    for i, img in enumerate(sampled_images):
                        img = img * 0.5 + 0.5
                        img = img.detach().cpu().permute(1, 2, 0).numpy()
                        img = (img * 255).astype(np.uint8)
                        img_pil = Image.fromarray(img)
                        img_pil.save(f'stable_diffusion_results/generated_image_epoch_{epoch+1}_sample_{i}.png')
                    wandb.log({f"generated_image_{i}": wandb.Image(sampled_images[i]) for i in range(len(sample_text))})
        # ====================================

    # ========== 保存最终模型（仅 rank=0）==========
    if rank == 0:
        torch.save(model.module.state_dict(), 'stable_diffusion_model_final.pth')
        wandb.finish()

    dist.destroy_process_group()

if __name__ == "__main__":
    main()