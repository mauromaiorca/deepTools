#!/usr/bin/env python
# train_GAN.py

import random
import math
import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from .dataset import CryoEMDataset
from .model_GAN import UNetWithAttention, Discriminator3D
from .utils import log_to_file, load_mrc_file
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

def split_directories(data_list, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, split_file=None):
    directories = list(set(os.path.dirname(entry['map_file']) for entry in data_list))
    random.shuffle(directories)
    num_train = int(train_ratio * len(directories))
    num_val = int(val_ratio * len(directories))
    train_dirs = directories[:num_train]
    val_dirs = directories[num_train:num_train + num_val]
    test_dirs = directories[num_train + num_val:]
    dir_split = {"train": train_dirs, "val": val_dirs, "test": test_dirs}
    if split_file:
        with open(split_file, 'w') as f:
            json.dump(dir_split, f, indent=4)
    return dir_split

def compute_global_robust_scaling_factor(samples, num_samples=50, epsilon=1e-8):
    ratios = []
    sample_subset = samples[:min(num_samples, len(samples))]
    for sample in sample_subset:
        file_path = sample["map_path"]
        volume = load_mrc_file(file_path)
        if volume is None:
            continue
        med = np.median(volume)
        mad = np.median(np.abs(volume - med))
        std = np.std(volume)
        if mad > epsilon:
            ratios.append(std / mad)
    if len(ratios) == 0:
        return 1.4826
    return np.median(ratios)

def select_indices_by_split(data_list, dir_split, split_type):
    split_dirs = dir_split[split_type]
    return [i for i, entry in enumerate(data_list) if os.path.dirname(entry['map_file']) in split_dirs]

def masked_loss(pred, target, mask, criterion=nn.MSELoss(reduction='none')):
    if mask is not None:
        loss = criterion(pred, target)
        masked_loss_val = loss * mask
        return masked_loss_val.sum() / (mask.sum() + 1e-8)
    else:
        return criterion(pred, target)

def custom_collate(batch):
    inputs, targets, masks = zip(*batch)
    masks = [m if m is not None else torch.zeros_like(targets[0], dtype=torch.bool) for m in masks]
    return torch.stack(inputs), torch.stack(targets), torch.stack(masks)

def main():
    parser = argparse.ArgumentParser(
        description="Training script for GAN-based cryo-EM image enhancement and parameter estimation.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--config", required=True, help="Path to configuration JSON file.")
    parser.add_argument("--model_file", required=True, help="Path to the generator model file (.pth) for saving or loading.")
    parser.add_argument("--disc_file", required=True, help="Path to the discriminator model file (.pth) for saving or loading.")
    parser.add_argument("--use_pretrained", action="store_true",
                        help="Flag to indicate using a pre-trained generator model. If not set, a new model will be initialised and overwrite the model file if it exists.")
    parser.add_argument("--training_data", required=True, help="Path to the training_data.json file.")
    parser.add_argument("--log", help="Path to log file.")
    parser.add_argument("--split_file", help="Path to save or load the train/val/test split information.")
    parser.add_argument("--g", default="0", help="Comma separated GPU ids, e.g. 0,1,2,3")
    args = parser.parse_args()

    print("### train.py started ###")
    print(f"Config file: {args.config}")
    print(f"Generator Model file: {args.model_file}")
    print(f"Discriminator Model file: {args.disc_file}")
    print(f"Training data: {args.training_data}")
    if args.log:
        print(f"Log file: {args.log}")

    with open(args.config, 'r') as cf:
        config = json.load(cf)
    with open(args.training_data, 'r') as f:
        data_list = json.load(f)

    if args.split_file and os.path.exists(args.split_file):
        with open(args.split_file, 'r') as f:
            dir_split = json.load(f)
    else:
        dir_split = split_directories(
            data_list,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            split_file=args.split_file
        )
    
    gpu_ids = [int(s) for s in args.g.split(',')]
    device = torch.device(f"cuda:{gpu_ids[0]}" if torch.cuda.is_available() else "cpu")

    train_indices = select_indices_by_split(data_list, dir_split, "train")
    val_indices = select_indices_by_split(data_list, dir_split, "val")
    test_indices = select_indices_by_split(data_list, dir_split, "test")

    if config.get("robust_normalization_auto_compute_scaling", False):
        global_scaling_factor = compute_global_robust_scaling_factor(data_list, num_samples=50)
    else:
        global_scaling_factor = 1.4826

    dataset = CryoEMDataset(
        data_list=data_list,
        cryo_em_keys=config["map_keys"]["cryo_em"],
        target_keys=config["map_keys"]["target"],
        mask_keys=config["map_keys"]["mask"] if config["map_keys"].get("mask") else None,
        scale_factor=config.get("scale_factor", 0.3),
        pixel_size_default=config.get("pixel_size_default", 1.0),
        normalize_input=config.get("normalization_input", True),
        use_robust_normalization=config.get("use_robust_normalization", False),
        normalize_reference=config.get("normalization_reference", False),
        max_volume_side_length=config.get("max_volume_side_length", -1),
        use_random_patch_training=config.get("use_random_patch_training", False),
        random_patch_training_size=config.get("random_patch_training_size", 64),
        masked_crop=config.get("masked_crop", False),
        inline_input_preProcessing=config.get("inline_input_preProcessing", False),
        inline_input_preProcessing_command=config.get("inline_input_preProcessing_command", "--threshold 0.01 --dilate 3 --close 5"),
        inline_target_preProcessing=config.get("inline_target_preProcessing", False),
        inline_target_preProcessing_command=config.get("inline_target_preProcessing_command", "--threshold 0.01 --dilate 3 --close 5"),
        masked_learning=config.get("masked_learning", False),
        logfile=args.log,
        random_rotation=config.get("random_rotation", False),
        robust_normalization_alpha=config.get("robust_normalization_alpha", 0.5),
        robust_fallback_scaling=1.0, 
        robust_smoothing=0.1
    )

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=config.get('batch_size', 1), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.get('batch_size', 1), shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.get('batch_size', 1), shuffle=False)

    print(f"Using device: {device}")
    if args.log:
        log_to_file(args.log, f"Training started with config: {config}")
        log_to_file(args.log, f"Using device: {device}")

    # Instantiate generator and discriminator
    generator = UNetWithAttention(in_channels=1, out_channels=1).to(device)
    discriminator = Discriminator3D(in_channels=1).to(device)

    if len(gpu_ids) > 1:
        generator = torch.nn.DataParallel(generator, device_ids=gpu_ids)
        discriminator = torch.nn.DataParallel(discriminator, device_ids=gpu_ids)

    if os.path.exists(args.model_file) and args.use_pretrained:
        print(f"Loading existing generator model from {args.model_file}")
        generator.load_state_dict(torch.load(args.model_file, map_location=device))
    else:
        print("Starting with a new generator model.")

    if os.path.exists(args.disc_file) and args.use_pretrained:
        print(f"Loading existing discriminator model from {args.disc_file}")
        discriminator.load_state_dict(torch.load(args.disc_file, map_location=device))
    else:
        print("Starting with a new discriminator model.")

    initial_lr = config.get('learning_rate_initial', 1e-4)
    scale_factor = config.get('scale_factor', 0.3)
    adjusted_lr = initial_lr / (scale_factor + 1e-8)
    optimizer_G = optim.Adam(generator.parameters(), lr=adjusted_lr)
    lr_disc = config.get('learning_rate_discriminator', adjusted_lr)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr_disc)

    scheduler_G = CosineAnnealingWarmRestarts(
        optimizer_G,
        T_0=config.get("learning_rate_warm_restart_initial_period", 10),
        T_mult=config.get("learning_rate_warm_restart_multiplier", 1),
        eta_min=config.get("learning_rate_minimum", 1e-6)
    )
    scheduler_D = CosineAnnealingWarmRestarts(
        optimizer_D,
        T_0=config.get("learning_rate_warm_restart_initial_period", 10),
        T_mult=config.get("learning_rate_warm_restart_multiplier", 1),
        eta_min=config.get("learning_rate_minimum", 1e-6)
    )

    scaler_G = GradScaler()
    scaler_D = GradScaler()

    bce_loss = nn.BCEWithLogitsLoss()
    lambda_recon = config.get("lambda_recon", 1.0)
    lambda_adv = config.get("lambda_adv", 0.001)
    num_epochs = config.get('num_epochs', 50)
    best_val_loss = float('inf')

    # Parameters for EMA smoothing of discriminator loss
    ema_alpha = 0.9
    running_d_loss = None
    gain = 0.1
    target_d_loss = 0.1

    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()
        train_loss_G = 0.0
        train_loss_D = 0.0
        valid_train_samples = 0

        for inputs, targets, masks in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            masks = masks.to(device) if masks is not None else None
            batch_size = inputs.size(0)

            # Update discriminator
            optimizer_D.zero_grad()
            with autocast():
                real_labels = torch.ones(batch_size, 1, device=device)
                fake_labels = torch.zeros(batch_size, 1, device=device)
                d_real = discriminator(targets)
                loss_real = bce_loss(d_real, real_labels)
                fake = generator(inputs).detach()
                d_fake = discriminator(fake)
                loss_fake = bce_loss(d_fake, fake_labels)
                loss_D = (loss_real + loss_fake) / 2
            scaler_D.scale(loss_D).backward()
            if config.get("enable_gradient_clipping", False):
                scaler_D.unscale_(optimizer_D)
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), config.get("gradient_clipping_max_norm", 1.0))
            scaler_D.step(optimizer_D)
            scaler_D.update()

            # Update generator
            optimizer_G.zero_grad()
            with autocast():
                fake = generator(inputs)
                recon_loss = masked_loss(fake, targets, masks)
                d_fake_for_G = discriminator(fake)
                adv_loss = bce_loss(d_fake_for_G, real_labels)
                loss_G = lambda_recon * recon_loss + lambda_adv * adv_loss
            scaler_G.scale(loss_G).backward()
            if config.get("enable_gradient_clipping", False):
                scaler_G.unscale_(optimizer_G)
                torch.nn.utils.clip_grad_norm_(generator.parameters(), config.get("gradient_clipping_max_norm", 1.0))
            scaler_G.step(optimizer_G)
            scaler_G.update()

            valid_train_samples += batch_size
            train_loss_G += loss_G.item() * batch_size
            train_loss_D += loss_D.item() * batch_size

        if valid_train_samples > 0:
            train_loss_G /= valid_train_samples
            train_loss_D /= valid_train_samples
        else:
            train_loss_G = float('nan')
            train_loss_D = float('nan')

        # Validation loop (evaluating the generator's reconstruction loss)
        generator.eval()
        val_loss = 0.0
        valid_val_samples = 0
        with torch.no_grad(), autocast():
            for inputs, targets, masks in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                masks = masks.to(device) if masks is not None else None
                batch_size = inputs.size(0)
                fake = generator(inputs)
                loss = masked_loss(fake, targets, masks)
                valid_val_samples += batch_size
                val_loss += loss.item() * batch_size
        if valid_val_samples > 0:
            val_loss /= valid_val_samples
        else:
            val_loss = float('nan')

        epoch_message = (f"Epoch {epoch+1}/{num_epochs}, "
                         f"Train G Loss: {train_loss_G:.18f}, "
                         f"Train D Loss: {train_loss_D:.18f}, "
                         f"Val Recon Loss: {val_loss:.18f}")
        print(epoch_message)
        if args.log:
            log_to_file(args.log, epoch_message)

        # -------------------------------
        # Dynamic adjustment of λ_adv with EMA smoothing of discriminator loss
        current_d_loss = train_loss_D if math.isfinite(train_loss_D) else target_d_loss
        if running_d_loss is None:
            running_d_loss = current_d_loss
        else:
            running_d_loss = ema_alpha * running_d_loss + (1 - ema_alpha) * current_d_loss

        # Clamp running_d_loss to avoid extreme values
        running_d_loss = max(0.0, min(running_d_loss, 10.0))
        error = target_d_loss - running_d_loss
        lambda_adv = max(1e-6, lambda_adv * (1 + gain * error))
        print(f"Adjusted lambda_adv for next epoch: {lambda_adv:.6f}")

        scheduler_G.step(epoch + 1)
        scheduler_D.step(epoch + 1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(generator.state_dict(), args.model_file)
            torch.save(discriminator.state_dict(), args.disc_file)
            print("Models saved.")
            if args.log:
                log_to_file(args.log, "Models saved.")

    # Testing phase
    generator.load_state_dict(torch.load(args.model_file, map_location=device))
    generator.eval()
    test_loss = 0.0
    valid_test_samples = 0
    with torch.no_grad(), autocast():
        for inputs, targets, masks in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            masks = masks.to(device) if masks is not None else None
            batch_size = inputs.size(0)
            fake = generator(inputs)
            loss = masked_loss(fake, targets, masks)
            valid_test_samples += batch_size
            test_loss += loss.item() * batch_size
    if valid_test_samples > 0:
        test_loss /= valid_test_samples
    else:
        test_loss = float('nan')
    print(f'Test Reconstruction Loss: {test_loss:.8f}')

if __name__ == '__main__':
    main()
