# latent_dataset.py
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class PrecomputedLatentDataset(Dataset):
    def __init__(self, latent_file, augment_latent=False):
        """
        latent_file: str, path to .pt file saved by precompute_latents.py
        augment_latent: bool, whether to apply noise/data augmentation on latents
        """
        data = torch.load(latent_file)
        self.latents = data["latents"]  # torch.Tensor [N, 4, 64, 64]
        self.texts = data["texts"]      # List[str]
        self.augment_latent = augment_latent

        # Optional: latent-space augmentation (e.g., add small noise)
        self.noise_std = 0.05

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        latent = self.latents[idx]  # [4, 64, 64]
        text = self.texts[idx]

        if self.augment_latent and self.noise_std > 0:
            noise = torch.randn_like(latent) * self.noise_std
            latent = latent + noise

        return {
            "latents": latent,  # no .to(device) here!
            "text": text
        }