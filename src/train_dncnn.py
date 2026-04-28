import hashlib
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import config as cfg
from dataset_loader import get_dataset
from dncnn_model import DnCNN
from image_utils import tensor_to_rgb_numpy, rgb_to_gray
from noise import apply_noise


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_stable_rng(base_seed: int, image_id: int, noise_type: str):
    key = f"{base_seed}|{image_id}|{noise_type}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    local_seed = int.from_bytes(digest[:8], byteorder="little", signed=False) % (2 ** 32)
    return np.random.default_rng(local_seed)


class DnCNNDataset(Dataset):
    def __init__(self, dataset, noise_types):
        self.dataset = dataset
        self.noise_types = noise_types

    def __len__(self):
        return len(self.dataset) * len(self.noise_types)

    def __getitem__(self, idx):
        base_idx = idx // len(self.noise_types)
        noise_idx = idx % len(self.noise_types)
        noise_type = self.noise_types[noise_idx]

        img_tensor, _ = self.dataset[base_idx]

        rgb = tensor_to_rgb_numpy(img_tensor)
        clean = rgb_to_gray(rgb)

        rng = make_stable_rng(cfg.RANDOM_SEED, base_idx, noise_type)
        noisy = apply_noise(clean, noise_type, cfg, rng)

        clean_t = torch.from_numpy(clean).unsqueeze(0).float()
        noisy_t = torch.from_numpy(noisy).unsqueeze(0).float()

        return noisy_t, clean_t, noise_type


def evaluate(model, loader, device):
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0

    with torch.no_grad():
        for noisy, clean, _ in loader:
            noisy = noisy.to(device)
            clean = clean.to(device)

            pred = model(noisy)
            loss = criterion(pred, clean)
            total_loss += loss.item() * noisy.size(0)

    return total_loss / len(loader.dataset)


def main():
    set_seed(cfg.RANDOM_SEED)

    cfg.DATA_DIR.mkdir(parents=True, exist_ok=True)
    cfg.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading train dataset...")
    train_ds, info = get_dataset(
        cfg.DATASET_FLAG,
        cfg.IMAGE_SIZE,
        "train",
        cfg.DATA_DIR,
        max_images=None
    )

    print("Loading validation dataset...")
    val_ds, _ = get_dataset(
        cfg.DATASET_FLAG,
        cfg.IMAGE_SIZE,
        "val",
        cfg.DATA_DIR,
        max_images=None
    )

    train_set = DnCNNDataset(train_ds, cfg.DN_TRAIN_NOISE_TYPES)
    val_set = DnCNNDataset(val_ds, cfg.DN_TRAIN_NOISE_TYPES)

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.DN_BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.DN_NUM_WORKERS,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=cfg.DN_BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.DN_NUM_WORKERS,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = DnCNN(in_channels=1).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.DN_LR)
    criterion = nn.MSELoss()

    best_val = float("inf")
    save_path = cfg.CHECKPOINT_DIR / cfg.DN_SAVE_NAME

    for epoch in range(cfg.DN_EPOCHS):
        model.train()
        running = 0.0

        for noisy, clean, _ in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.DN_EPOCHS}"):
            noisy = noisy.to(device)
            clean = clean.to(device)

            optimizer.zero_grad()
            pred = model(noisy)
            loss = criterion(pred, clean)
            loss.backward()
            optimizer.step()

            running += loss.item() * noisy.size(0)

        train_loss = running / len(train_loader.dataset)
        val_loss = evaluate(model, val_loader, device)

        print(f"Epoch {epoch + 1}: train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to: {save_path}")

    print("Training finished.")
    print(f"Best checkpoint saved at: {save_path}")


if __name__ == "__main__":
    main()