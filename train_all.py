# train_all.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm  # For progress bars
import random
import time

# Import models
from models.modules import TargetRadialLengthModule, TargetIdentityModule
from models.cgan_models import Generator, Discriminator
from models.cae_models import ConvAutoEncoder
from models.msae_models import ModifiedSparseAutoEncoder
from models.msae_loss import MSAELoss
from utils.hrrp_dataset import HRRPDataset
from utils.noise_utils import add_noise_for_psnr


def train_feature_extractors(args, device, psnr_level=None):
    """
    Train the Target Radial Length Module (G_D) and Target Identity Module (G_I)

    Args:
        args: Training arguments
        device: Device to train on (CPU or GPU)
        psnr_level: If set, includes this in the output directory path

    Returns:
        Path to the directory where models are saved
    """
    print(f"Training feature extractors...")

    # Create output directory
    if psnr_level is not None:
        output_dir = os.path.join(args.output_dir, f"feature_extractors_psnr_{psnr_level}dB")
    else:
        output_dir = os.path.join(args.output_dir, "feature_extractors")
    os.makedirs(output_dir, exist_ok=True)

    # Create G_D model
    G_D = TargetRadialLengthModule(input_dim=args.input_dim, feature_dim=args.feature_dim).to(device)

    # Define loss function and optimizer for G_D
    criterion_GD = nn.MSELoss()
    optimizer_GD = optim.Adam(G_D.parameters(), lr=args.lr)

    # Create G_I model
    G_I = TargetIdentityModule(input_dim=args.input_dim, feature_dim=args.feature_dim,
                               num_classes=args.num_classes).to(device)

    # Define loss function and optimizer for G_I
    criterion_GI = nn.CrossEntropyLoss()
    optimizer_GI = optim.Adam(G_I.parameters(), lr=args.lr)

    # Load dataset
    train_dataset = HRRPDataset(args.train_dir)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Update num_classes based on dataset
    num_classes = train_dataset.get_num_classes()
    if num_classes != args.num_classes:
        print(f"Updating num_classes from {args.num_classes} to {num_classes} based on dataset")
        args.num_classes = num_classes
        G_I = TargetIdentityModule(input_dim=args.input_dim, feature_dim=args.feature_dim,
                                   num_classes=args.num_classes).to(device)
        optimizer_GI = optim.Adam(G_I.parameters(), lr=args.lr)

    # Print some info about the dataset
    print(f"Training feature extractors with dataset from: {args.train_dir}")
    print(f"Number of samples: {len(train_dataset)}")
    print(f"Number of classes: {args.num_classes}")

    # Tracking metrics
    gd_losses = []
    gi_losses = []
    gi_accuracies = []

    # Training loop
    for epoch in range(args.epochs):
        epoch_gd_loss = 0.0
        epoch_gi_loss = 0.0
        correct = 0
        total = 0

        # Progress bar for this epoch
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for data, radial_length, identity_labels in progress_bar:
            # Move data to device
            data = data.float().to(device)
            radial_length = radial_length.float().to(device)
            identity_labels = identity_labels.long().to(device)

            # ======== Train G_D ========
            optimizer_GD.zero_grad()
            _, predicted_radial_length = G_D(data)

            # Skip samples with invalid radial length
            valid_indices = ~torch.isnan(radial_length) & ~torch.isinf(radial_length) & (radial_length < 1e6)
            if valid_indices.sum() > 0:
                # Use only valid values to compute loss
                gd_loss = criterion_GD(
                    predicted_radial_length[valid_indices],
                    radial_length[valid_indices]
                )

                if not torch.isnan(gd_loss) and not torch.isinf(gd_loss) and gd_loss < 1e6:
                    gd_loss.backward()
                    optimizer_GD.step()
                    epoch_gd_loss += gd_loss.item()

            # ======== Train G_I ========
            optimizer_GI.zero_grad()
            _, identity_logits = G_I(data)
            gi_loss = criterion_GI(identity_logits, identity_labels)
            gi_loss.backward()
            optimizer_GI.step()
            epoch_gi_loss += gi_loss.item()

            # Calculate accuracy
            _, predicted = torch.max(identity_logits.data, 1)
            total += identity_labels.size(0)
            correct += (predicted == identity_labels).sum().item()

            # Update progress bar
            progress_bar.set_postfix({
                'G_D Loss': f"{gd_loss.item():.4f}" if 'gd_loss' in locals() else "N/A",
                'G_I Loss': f"{gi_loss.item():.4f}",
                'G_I Acc': f"{100 * correct / total:.2f}%"
            })

        # Calculate epoch metrics
        epoch_gd_loss /= len(train_loader)
        epoch_gi_loss /= len(train_loader)
        epoch_gi_accuracy = 100 * correct / total

        # Record metrics
        gd_losses.append(epoch_gd_loss)
        gi_losses.append(epoch_gi_loss)
        gi_accuracies.append(epoch_gi_accuracy)

        print(f"Epoch {epoch + 1}/{args.epochs} - "
              f"G_D Loss: {epoch_gd_loss:.4f}, "
              f"G_I Loss: {epoch_gi_loss:.4f}, "
              f"G_I Accuracy: {epoch_gi_accuracy:.2f}%")

        # Save checkpoints periodically
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_dir = os.path.join(output_dir, f"epoch_{epoch + 1}")
            os.makedirs(checkpoint_dir, exist_ok=True)

            torch.save(G_D.state_dict(), os.path.join(checkpoint_dir, 'G_D.pth'))
            torch.save(G_I.state_dict(), os.path.join(checkpoint_dir, 'G_I.pth'))

    # Save final models
    torch.save(G_D.state_dict(), os.path.join(output_dir, 'G_D_final.pth'))
    torch.save(G_I.state_dict(), os.path.join(output_dir, 'G_I_final.pth'))

    # Plot loss curves
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(gd_losses, label='G_D Loss')
    plt.plot(gi_losses, label='G_I Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Feature Extractors Training Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(gi_accuracies, label='G_I Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('G_I Classification Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'))
    plt.close()

    print(f"Feature extractors training complete. Models saved to {output_dir}")
    return output_dir


def train_cgan(args, device, psnr_level):
    """
    Train CGAN for HRRP signal denoising at a specific PSNR level

    Args:
        args: Training arguments
        device: Device to train on (CPU or GPU)
        psnr_level: Target PSNR level in dB

    Returns:
        Path to the directory where the model is saved
    """
    print(f"Training CGAN for PSNR level {psnr_level}dB...")

    # Create output directory
    output_dir = os.path.join(args.output_dir, f"cgan_psnr_{psnr_level}dB")
    os.makedirs(output_dir, exist_ok=True)

    # Load feature extractors
    G_D = TargetRadialLengthModule(input_dim=args.input_dim, feature_dim=args.feature_dim).to(device)
    G_I = TargetIdentityModule(input_dim=args.input_dim, feature_dim=args.feature_dim,
                               num_classes=args.num_classes).to(device)

    # Load their weights
    if args.feature_extractors_dir:
        G_D.load_state_dict(torch.load(os.path.join(args.feature_extractors_dir, 'G_D_final.pth')))
        G_I.load_state_dict(torch.load(os.path.join(args.feature_extractors_dir, 'G_I_final.pth')))

    # Create CGAN models
    generator = Generator(input_dim=args.input_dim,
                          condition_dim=args.feature_dim * 2,
                          hidden_dim=args.hidden_dim).to(device)

    discriminator = Discriminator(input_dim=args.input_dim,
                                  condition_dim=args.feature_dim * 2,
                                  hidden_dim=args.hidden_dim).to(device)

    # Define loss functions
    adversarial_loss = nn.BCELoss()
    reconstruction_loss = nn.MSELoss()
    regression_loss = nn.MSELoss()  # for G_D
    classification_loss = nn.CrossEntropyLoss()  # for G_I

    # Define optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_GD = optim.Adam(G_D.parameters(), lr=args.lr_feature_extractors, betas=(0.5, 0.999))
    optimizer_GI = optim.Adam(G_I.parameters(), lr=args.lr_feature_extractors, betas=(0.5, 0.999))

    # Load dataset
    train_dataset = HRRPDataset(args.train_dir)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Update num_classes based on dataset
    num_classes = train_dataset.get_num_classes()
    if num_classes != args.num_classes:
        print(f"Updating num_classes from {args.num_classes} to {num_classes} based on dataset")
        args.num_classes = num_classes
        G_I = TargetIdentityModule(input_dim=args.input_dim, feature_dim=args.feature_dim,
                                   num_classes=args.num_classes).to(device)
        if args.feature_extractors_dir:
            G_I.load_state_dict(torch.load(os.path.join(args.feature_extractors_dir, 'G_I_final.pth')))
        optimizer_GI = optim.Adam(G_I.parameters(), lr=args.lr_feature_extractors, betas=(0.5, 0.999))

    # Training statistics
    d_losses = []
    g_losses = []
    rec_losses = []
    gd_losses = []
    gi_losses = []

    # Training loop
    for epoch in range(args.epochs):
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        epoch_rec_loss = 0.0
        epoch_gd_loss = 0.0
        epoch_gi_loss = 0.0

        # Create progress bar
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for i, (clean_data, radial_length, identity_labels) in enumerate(progress_bar):
            batch_size = clean_data.shape[0]

            # Move data to device
            clean_data = clean_data.float().to(device)
            radial_length = radial_length.float().to(device)
            identity_labels = identity_labels.long().to(device)

            # Create noisy data at the target PSNR
            noisy_data = add_noise_for_psnr(clean_data, psnr_level)

            # ========================
            # 1. Extract features
            # ========================
            with torch.no_grad():
                f_D, _ = G_D(clean_data)
                f_I, _ = G_I(clean_data)
                condition = torch.cat([f_D, f_I], dim=1)

            # ========================
            # 2. Train Discriminator
            # ========================
            for _ in range(args.n_critic):
                optimizer_D.zero_grad()

                # Generate fake samples
                with torch.no_grad():
                    generated_samples = generator(noisy_data, condition)

                # Create labels with smoothing
                real_labels = torch.ones(batch_size, 1).to(device) * 0.9
                fake_labels = torch.zeros(batch_size, 1).to(device) + 0.1

                # Discriminator loss for real samples
                real_pred = discriminator(clean_data, condition)
                real_loss = adversarial_loss(real_pred, real_labels)

                # Discriminator loss for fake samples
                fake_pred = discriminator(generated_samples.detach(), condition)
                fake_loss = adversarial_loss(fake_pred, fake_labels)

                # Total discriminator loss
                d_loss = (real_loss + fake_loss) / 2

                # Gradient penalty (optional)
                if args.use_gp:
                    alpha = torch.rand(batch_size, 1).to(device)
                    interpolated = (alpha * clean_data + (1 - alpha) * generated_samples.detach()).requires_grad_(True)
                    interp_condition = condition.detach()

                    d_interp = discriminator(interpolated, interp_condition)

                    gradients = torch.autograd.grad(
                        outputs=d_interp,
                        inputs=interpolated,
                        grad_outputs=torch.ones_like(d_interp),
                        create_graph=True,
                        retain_graph=True,
                    )[0]

                    gradients = gradients.view(batch_size, -1)
                    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
                    d_loss = d_loss + args.lambda_gp * gradient_penalty

                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), args.clip_value)
                optimizer_D.step()

            # ========================
            # 3. Train Generator
            # ========================
            optimizer_G.zero_grad()

            # Generate samples
            generated_samples = generator(noisy_data, condition)

            # Adversarial loss (fool the discriminator)
            g_adv_loss = adversarial_loss(discriminator(generated_samples, condition), real_labels)

            # Reconstruction loss
            g_rec_loss = reconstruction_loss(generated_samples, clean_data)

            # Total generator loss
            g_loss = g_adv_loss + args.lambda_rec * g_rec_loss
            g_loss.backward()

            torch.nn.utils.clip_grad_norm_(generator.parameters(), args.clip_value)
            optimizer_G.step()

            # ========================
            # 4. Update G_D (optional)
            # ========================
            if args.update_feature_extractors:
                optimizer_GD.zero_grad()

                # Predict radial length
                _, pred_radial = G_D(clean_data)

                # Skip samples with invalid radial length
                valid_indices = ~torch.isnan(radial_length) & ~torch.isinf(radial_length) & (radial_length < 1e6)
                if valid_indices.sum() > 0:
                    # Use only valid values to compute loss
                    gd_loss = regression_loss(
                        pred_radial[valid_indices],
                        radial_length[valid_indices]
                    )

                    if not torch.isnan(gd_loss) and not torch.isinf(gd_loss) and gd_loss < 1e6:
                        (args.lambda_gd * gd_loss).backward()
                        torch.nn.utils.clip_grad_norm_(G_D.parameters(), args.clip_value)
                        optimizer_GD.step()
                        epoch_gd_loss += gd_loss.item()

                # ========================
                # 5. Update G_I (optional)
                # ========================
                optimizer_GI.zero_grad()

                # Predict identity
                _, pred_identity = G_I(clean_data)

                # Calculate G_I classification loss
                gi_loss = classification_loss(pred_identity, identity_labels)

                # Apply loss weight and backpropagate
                (args.lambda_gi * gi_loss).backward()

                torch.nn.utils.clip_grad_norm_(G_I.parameters(), args.clip_value)
                optimizer_GI.step()

                epoch_gi_loss += gi_loss.item()

            # Update epoch losses
            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_adv_loss.item()
            epoch_rec_loss += g_rec_loss.item()

            # Update progress bar
            progress_bar.set_postfix({
                'D_Loss': f"{d_loss.item():.4f}",
                'G_Adv': f"{g_adv_loss.item():.4f}",
                'G_Rec': f"{g_rec_loss.item():.4f}"
            })

        # Calculate average losses for the epoch
        epoch_d_loss /= len(train_loader)
        epoch_g_loss /= len(train_loader)
        epoch_rec_loss /= len(train_loader)
        epoch_gd_loss /= len(train_loader)
        epoch_gi_loss /= len(train_loader)

        # Save losses for plotting
        d_losses.append(epoch_d_loss)
        g_losses.append(epoch_g_loss)
        rec_losses.append(epoch_rec_loss)
        gd_losses.append(epoch_gd_loss)
        gi_losses.append(epoch_gi_loss)

        print(f"Epoch {epoch + 1}/{args.epochs} - "
              f"D: {epoch_d_loss:.4f}, G_adv: {epoch_g_loss:.4f}, G_rec: {epoch_rec_loss:.4f}, "
              f"G_D: {epoch_gd_loss:.4f}, G_I: {epoch_gi_loss:.4f}")

        # Save checkpoint periodically
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_dir = os.path.join(output_dir, f"epoch_{epoch + 1}")
            os.makedirs(checkpoint_dir, exist_ok=True)

            # Save all models
            torch.save(generator.state_dict(), os.path.join(checkpoint_dir, 'generator.pth'))
            torch.save(discriminator.state_dict(), os.path.join(checkpoint_dir, 'discriminator.pth'))
            if args.update_feature_extractors:
                torch.save(G_D.state_dict(), os.path.join(checkpoint_dir, 'G_D.pth'))
                torch.save(G_I.state_dict(), os.path.join(checkpoint_dir, 'G_I.pth'))

            # Generate and save a sample denoised image
            if args.save_samples:
                with torch.no_grad():
                    # Get a sample from the dataset
                    sample_idx = np.random.randint(0, len(train_dataset))
                    sample_clean, _, _ = train_dataset[sample_idx]
                    sample_clean = sample_clean.unsqueeze(0).float().to(device)

                    # Create noisy sample at the target PSNR
                    sample_noisy = add_noise_for_psnr(sample_clean, psnr_level)

                    # Extract features
                    f_D, _ = G_D(sample_clean)
                    f_I, _ = G_I(sample_clean)
                    sample_condition = torch.cat([f_D, f_I], dim=1)

                    # Generate denoised sample
                    sample_denoised = generator(sample_noisy, sample_condition)

                    # Plot and save results
                    plt.figure(figsize=(12, 4))

                    plt.subplot(1, 3, 1)
                    plt.plot(sample_clean.cpu().numpy()[0])
                    plt.title('Clean HRRP')
                    plt.xlabel('Range')
                    plt.ylabel('Magnitude')

                    plt.subplot(1, 3, 2)
                    plt.plot(sample_noisy.cpu().numpy()[0])
                    plt.title(f'Noisy HRRP (PSNR={psnr_level}dB)')
                    plt.xlabel('Range')
                    plt.ylabel('Magnitude')

                    plt.subplot(1, 3, 3)
                    plt.plot(sample_denoised.cpu().numpy()[0])
                    plt.title('Denoised HRRP')
                    plt.xlabel('Range')
                    plt.ylabel('Magnitude')

                    plt.tight_layout()
                    plt.savefig(os.path.join(checkpoint_dir, 'sample_denoising.png'))
                    plt.close()

    # Save final models
    torch.save(generator.state_dict(), os.path.join(output_dir, 'generator_final.pth'))
    torch.save(discriminator.state_dict(), os.path.join(output_dir, 'discriminator_final.pth'))
    if args.update_feature_extractors:
        torch.save(G_D.state_dict(), os.path.join(output_dir, 'G_D_final.pth'))
        torch.save(G_I.state_dict(), os.path.join(output_dir, 'G_I_final.pth'))

    # Plot loss curves
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.plot(d_losses, label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Discriminator Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(g_losses, label='Generator Adversarial Loss')
    plt.plot(rec_losses, label='Generator Reconstruction Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Generator Losses')
    plt.legend()
    plt.grid(True)

    if args.update_feature_extractors:
        plt.subplot(2, 2, 3)
        plt.plot(gd_losses, label='G_D Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('G_D Loss')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 2, 4)
        plt.plot(gi_losses, label='G_I Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('G_I Loss')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_loss.png'))
    plt.close()

    print(f"CGAN training complete for PSNR={psnr_level}dB. Models saved to {output_dir}")
    return output_dir


def train_cae(args, device, psnr_level):
    """
    Train CAE for HRRP signal denoising at a specific PSNR level

    Args:
        args: Training arguments
        device: Device to train on (CPU or GPU)
        psnr_level: Target PSNR level in dB

    Returns:
        Path to the directory where the model is saved
    """
    print(f"Training CAE for PSNR level {psnr_level}dB...")

    # Create output directory
    output_dir = os.path.join(args.output_dir, f"cae_psnr_{psnr_level}dB")
    os.makedirs(output_dir, exist_ok=True)

    # Create CAE model
    model = ConvAutoEncoder(input_dim=args.input_dim,
                            latent_dim=args.latent_dim,
                            hidden_dim=args.hidden_dim).to(device)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Load dataset
    train_dataset = HRRPDataset(args.train_dir)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Print information about the dataset and model
    print(f"Training CAE with dataset from: {args.train_dir}")
    print(f"Number of samples: {len(train_dataset)}")

    # Arrays for tracking loss
    train_losses = []

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0

        # Create progress bar
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for i, (clean_data, _, _) in enumerate(progress_bar):
            # Move data to device
            clean_data = clean_data.float().to(device)

            # Create noisy data at the target PSNR
            noisy_data = add_noise_for_psnr(clean_data, psnr_level)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass (denoising task: noisy->clean)
            reconstructed, _ = model(noisy_data)

            # Calculate loss (between reconstruction and clean data)
            loss = criterion(reconstructed, clean_data)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Update epoch loss
            epoch_loss += loss.item()

            # Update progress bar
            progress_bar.set_postfix({'Loss': f"{loss.item():.4f}"})

        # Calculate average epoch loss
        epoch_loss /= len(train_loader)
        train_losses.append(epoch_loss)

        # Print epoch summary
        print(f"Epoch {epoch + 1}/{args.epochs} - Avg Loss: {epoch_loss:.4f}")

        # Save checkpoint periodically
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_dir = os.path.join(output_dir, f"epoch_{epoch + 1}")
            os.makedirs(checkpoint_dir, exist_ok=True)

            # Save model
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'cae_model.pth'))

            # Save sample denoising results
            if args.save_samples:
                model.eval()
                with torch.no_grad():
                    # Get sample from dataset
                    sample_idx = np.random.randint(0, len(train_dataset))
                    sample_clean, _, _ = train_dataset[sample_idx]
                    sample_clean = sample_clean.unsqueeze(0).float().to(device)

                    # Create noisy sample at the target PSNR
                    sample_noisy = add_noise_for_psnr(sample_clean, psnr_level)

                    # Denoise the sample
                    sample_denoised, _ = model(sample_noisy)

                    # Plot results
                    plt.figure(figsize=(12, 4))

                    plt.subplot(1, 3, 1)
                    plt.plot(sample_clean.cpu().numpy()[0])
                    plt.title('Clean HRRP')
                    plt.xlabel('Range')
                    plt.ylabel('Magnitude')

                    plt.subplot(1, 3, 2)
                    plt.plot(sample_noisy.cpu().numpy()[0])
                    plt.title(f'Noisy HRRP (PSNR={psnr_level}dB)')
                    plt.xlabel('Range')
                    plt.ylabel('Magnitude')

                    plt.subplot(1, 3, 3)
                    plt.plot(sample_denoised.cpu().numpy()[0])
                    plt.title('Denoised HRRP')
                    plt.xlabel('Range')
                    plt.ylabel('Magnitude')

                    plt.tight_layout()
                    plt.savefig(os.path.join(checkpoint_dir, 'sample_denoising.png'))
                    plt.close()

                model.train()

    # Save final model
    torch.save(model.state_dict(), os.path.join(output_dir, 'cae_model_final.pth'))

    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'CAE Training Loss (PSNR={psnr_level}dB)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'training_loss.png'))
    plt.close()

    print(f"CAE training complete for PSNR={psnr_level}dB. Model saved to {output_dir}")
    return output_dir


# train_msae function to be added to train_all.py
def train_msae(args, device, psnr_level):
    """
    Train Modified Sparse AutoEncoder for HRRP signal denoising at a specific PSNR level

    Args:
        args: Training arguments
        device: Device to train on (CPU or GPU)
        psnr_level: Target PSNR level in dB

    Returns:
        Path to the directory where the model is saved
    """
    print(f"Training MSAE for PSNR level {psnr_level}dB...")

    # Create output directory
    output_dir = os.path.join(args.output_dir, f"msae_psnr_{psnr_level}dB")
    os.makedirs(output_dir, exist_ok=True)

    # Create MSAE model
    model = ModifiedSparseAutoEncoder(
        input_dim=args.input_dim,
        latent_dim=args.latent_dim,
        hidden_dim=args.msae_hidden_dim,
        sparsity_param=args.sparsity_param,
        reg_lambda=args.reg_lambda,
        sparsity_beta=args.sparsity_beta
    ).to(device)

    # Define loss function and optimizer
    criterion = MSAELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Load dataset
    train_dataset = HRRPDataset(args.train_dir)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Print information about the dataset and model
    print(f"Training MSAE with dataset from: {args.train_dir}")
    print(f"Number of samples: {len(train_dataset)}")
    print(f"Sparsity parameter: {args.sparsity_param}, Beta: {args.sparsity_beta}")

    # Arrays for tracking loss
    train_losses = []
    reconstruction_losses = []
    weight_reg_losses = []
    sparsity_losses = []

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_rec_loss = 0.0
        epoch_weight_loss = 0.0
        epoch_sparsity_loss = 0.0

        # Create progress bar
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for i, (clean_data, _, _) in enumerate(progress_bar):
            # Move data to device
            clean_data = clean_data.float().to(device)

            # Create noisy data at the target PSNR
            noisy_data = add_noise_for_psnr(clean_data, psnr_level)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass (denoising task: noisy->clean)
            reconstructed, latent = model(noisy_data)

            # Calculate loss
            loss, loss_components = criterion(model, clean_data, reconstructed, latent)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Update epoch loss
            epoch_loss += loss_components['total']
            epoch_rec_loss += loss_components['reconstruction']
            epoch_weight_loss += loss_components['weight_reg']
            epoch_sparsity_loss += loss_components['sparsity']

            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f"{loss_components['total']:.4f}",
                'Rec': f"{loss_components['reconstruction']:.4f}",
                'Sparsity': f"{loss_components['sparsity']:.4f}"
            })

        # Calculate average epoch loss
        epoch_loss /= len(train_loader)
        epoch_rec_loss /= len(train_loader)
        epoch_weight_loss /= len(train_loader)
        epoch_sparsity_loss /= len(train_loader)

        # Save losses for plotting
        train_losses.append(epoch_loss)
        reconstruction_losses.append(epoch_rec_loss)
        weight_reg_losses.append(epoch_weight_loss)
        sparsity_losses.append(epoch_sparsity_loss)

        # Print epoch summary
        print(f"Epoch {epoch + 1}/{args.epochs} - "
              f"Avg Loss: {epoch_loss:.4f}, Rec: {epoch_rec_loss:.4f}, "
              f"Weight: {epoch_weight_loss:.4f}, Sparsity: {epoch_sparsity_loss:.4f}")

        # Apply SVD weight modification periodically
        if (epoch + 1) % args.svd_interval == 0:
            model.modify_weights_with_svd(threshold=args.svd_threshold)
            print(f"Applied SVD weight modification at epoch {epoch + 1}")

        # Save checkpoint periodically
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_dir = os.path.join(output_dir, f"epoch_{epoch + 1}")
            os.makedirs(checkpoint_dir, exist_ok=True)

            # Save model
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'msae_model.pth'))

            # Save sample denoising results
            if args.save_samples:
                model.eval()
                with torch.no_grad():
                    # Get sample from dataset
                    sample_idx = np.random.randint(0, len(train_dataset))
                    sample_clean, _, _ = train_dataset[sample_idx]
                    sample_clean = sample_clean.unsqueeze(0).float().to(device)

                    # Create noisy sample at the target PSNR
                    sample_noisy = add_noise_for_psnr(sample_clean, psnr_level)

                    # Denoise the sample
                    sample_denoised, _ = model(sample_noisy)

                    # Plot results
                    plt.figure(figsize=(12, 4))

                    plt.subplot(1, 3, 1)
                    plt.plot(sample_clean.cpu().numpy()[0])
                    plt.title('Clean HRRP')
                    plt.xlabel('Range')
                    plt.ylabel('Magnitude')

                    plt.subplot(1, 3, 2)
                    plt.plot(sample_noisy.cpu().numpy()[0])
                    plt.title(f'Noisy HRRP (PSNR={psnr_level}dB)')
                    plt.xlabel('Range')
                    plt.ylabel('Magnitude')

                    plt.subplot(1, 3, 3)
                    plt.plot(sample_denoised.cpu().numpy()[0])
                    plt.title('MSAE Denoised HRRP')
                    plt.xlabel('Range')
                    plt.ylabel('Magnitude')

                    plt.tight_layout()
                    plt.savefig(os.path.join(checkpoint_dir, 'sample_denoising.png'))
                    plt.close()

                model.train()

    # Save final model
    torch.save(model.state_dict(), os.path.join(output_dir, 'msae_model_final.pth'))

    # Plot loss curves
    plt.figure(figsize=(15, 6))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Total Loss')
    plt.plot(reconstruction_losses, label='Reconstruction Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'MSAE Training Loss (PSNR={psnr_level}dB)')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(weight_reg_losses, label='Weight Regularization')
    plt.plot(sparsity_losses, label='Sparsity Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Regularization Components')
    plt.legend()
    plt.grid(True)

    plt.savefig(os.path.join(output_dir, 'training_loss.png'))
    plt.close()

    print(f"MSAE training complete for PSNR={psnr_level}dB. Model saved to {output_dir}")
    return output_dir


def main():
    # Define command-line arguments
    parser = argparse.ArgumentParser(description='Unified training script for HRRP denoising models')

    # General parameters
    parser.add_argument('--model', type=str, default='all',
                        choices=['feature_extractors', 'cgan', 'cae', 'msae', 'all'],
                        help='Model to train')
    parser.add_argument('--train_dir', type=str, default='datasets/simulated_3/train',
                        help='Directory containing training data')
    parser.add_argument('--output_dir', type=str, default='checkpoints',
                        help='Directory to save trained models')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.00001,
                        help='Learning rate')
    parser.add_argument('--input_dim', type=int, default=500,
                        help='Dimension of input HRRP sequence')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='Epoch interval for saving checkpoints')
    parser.add_argument('--save_samples', action='store_true',
                        help='Whether to save sample denoising results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--psnr_levels', type=str, default='20,10,5',
                        help='PSNR levels to train at (comma-separated values in dB)')

    # Feature extractors parameters
    parser.add_argument('--feature_dim', type=int, default=64,
                        help='Dimension of feature extractors output')
    parser.add_argument('--num_classes', type=int, default=3,
                        help='Number of target identity classes')

    # CGAN specific parameters
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Dimension of hidden layers for CGAN and CAE')
    parser.add_argument('--feature_extractors_dir', type=str, default='checkpoints/feature_extractors',
                        help='Directory containing pre-trained feature extractors')
    parser.add_argument('--lr_feature_extractors', type=float, default=0.00001,
                        help='Learning rate for fine-tuning feature extractors')
    parser.add_argument('--lambda_rec', type=float, default=10.0,
                        help='Weight of reconstruction loss in CGAN')
    parser.add_argument('--lambda_gd', type=float, default=0.0001,
                        help='Weight of G_D regression loss')
    parser.add_argument('--lambda_gi', type=float, default=0.1,
                        help='Weight of G_I classification loss')
    parser.add_argument('--lambda_gp', type=float, default=10.0,
                        help='Weight of gradient penalty')
    parser.add_argument('--n_critic', type=int, default=1,
                        help='Number of discriminator updates per generator update')
    parser.add_argument('--use_gp', action='store_true',
                        help='Whether to use gradient penalty')
    parser.add_argument('--clip_value', type=float, default=1.0,
                        help='Gradient clipping value')
    parser.add_argument('--update_feature_extractors', default=1,
                        help='Whether to update feature extractors during CGAN training')

    # CAE and MSAE specific parameters
    parser.add_argument('--latent_dim', type=int, default=64,
                        help='Dimension of latent space for CAE and AE')
    parser.add_argument('--msae_hidden_dim', type=int, default=128,
                        help='Dimension of hidden layers for MSAE')
    parser.add_argument('--sparsity_param', type=float, default=0.05,
                        help='Sparsity parameter (p) for MSAE')
    parser.add_argument('--reg_lambda', type=float, default=0.0001,
                        help='Weight regularization parameter (lambda) for MSAE')
    parser.add_argument('--sparsity_beta', type=float, default=3.0,
                        help='Sparsity weight parameter (beta) for MSAE')
    parser.add_argument('--svd_interval', type=int, default=10,
                        help='Interval for SVD weight modification in MSAE')
    parser.add_argument('--svd_threshold', type=float, default=0.1,
                        help='Threshold for singular value pruning in MSAE')

    args = parser.parse_args()

    # Parse PSNR levels
    psnr_levels = [float(level) for level in args.psnr_levels.split(',')]

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save training configuration
    with open(os.path.join(args.output_dir, 'training_config.txt'), 'w') as f:
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")

    # Train the selected models
    if args.model in ['feature_extractors', 'all']:
        feature_extractors_dir = train_feature_extractors(args, device)
        args.feature_extractors_dir = feature_extractors_dir

    # Train models for each PSNR level
    for psnr_level in psnr_levels:
        print(f"\n{'=' * 50}")
        print(f"Training for PSNR level: {psnr_level}dB")
        print(f"{'=' * 50}\n")

        if args.model in ['cgan', 'all']:
            train_cgan(args, device, psnr_level)

        if args.model in ['cae', 'all']:
            train_cae(args, device, psnr_level)

        if args.model in ['msae', 'all']:
            train_msae(args, device, psnr_level)

    print("\nTraining complete for all models and PSNR levels.")


if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Total training time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")