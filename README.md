# Denoising HRRP with Conditional GAN

This repository implements a Conditional Generative Adversarial Network (CGAN) approach for High-Resolution Range Profile (HRRP) signal denoising. The implementation uses deep learning techniques to remove noise from HRRP signals while preserving important target characteristics.

## Overview

High-Resolution Range Profiles (HRRPs) are often contaminated with noise during acquisition, which can degrade radar system performance. This project leverages conditional generative adversarial networks to learn the mapping between noisy and clean HRRP signals, with conditioning on target identity and radial length information.

## Repository Structure

```
├── .gitignore
├── .idea/
├── LICENSE.txt
├── README.md
├── __pycache__/
├── checkpoints/
│   └── G_D.pth
├── hrrp_dataset.py
├── models.py
├── test_modules.py
├── train_modules.py
├── cgan_models.py      # CGAN generator and discriminator models
├── train_cgan.py       # CGAN adversarial training implementation
└── test_cgan.py        # Evaluation of trained CGAN models
```

## Requirements

- Python 3.8
- PyTorch 2.2.0
- CUDA 12.5
- NumPy
- Matplotlib
- SciPy

## Implementation Details

### Models

1. **Feature Extractors**:
   - `TargetRadialLengthModule (G_D)`: Extracts radial length information from HRRP signals
   - `TargetIdentityModule (G_I)`: Extracts target identity information from HRRP signals

2. **CGAN Architecture**:
   - `Generator`: 1D CNN-based generator that transforms noisy HRRP to clean HRRP
   - `Discriminator`: 1D CNN-based discriminator that distinguishes between real and generated clean HRRP signals

### Adversarial Training Process

The CGAN training process consists of the following steps:

1. Extract conditioning features from clean HRRP signals using pre-trained G_D and G_I modules
2. Add Gaussian noise to create noisy HRRP signals
3. Train the discriminator to classify real vs. generated samples
4. Train the generator to produce denoised signals that fool the discriminator
5. Apply both adversarial loss and reconstruction loss to optimize the generator
6. Periodically save checkpoints and sample denoising results

## Dataset

The implementation expects HRRP data in MATLAB (.mat) format organized in a directory structure. Each HRRP sample should contain:
- The HRRP signal data stored as 'CoHH' variable in the .mat file
- Target identity information encoded in the filename
- Radial length information encoded in the filename

## Usage

### Training the Feature Extractors

Before training the CGAN, train the feature extractors if they're not already available:

```bash
python train_modules.py --module both --train_dir datasets/simulated_3/train --save_dir checkpoints --batch_size 256 --epochs 1000
```

### Training the CGAN

```bash
python train_cgan.py --train_dir datasets/simulated_3/train --load_feature_extractors --load_dir checkpoints --save_dir checkpoints/cgan --batch_size 64 --epochs 200 --noise_level 0.1 --lambda_rec 10.0 --save_samples
```

Key parameters:
- `--train_dir`: Directory containing training data
- `--load_feature_extractors`: Flag to load pre-trained feature extractors
- `--noise_level`: Standard deviation of Gaussian noise to add
- `--lambda_rec`: Weight of reconstruction loss in generator loss function
- `--save_samples`: Save sample denoising results during training

### Testing the CGAN

```bash
python test_cgan.py --test_dir datasets/simulated_3/test --load_dir checkpoints/cgan --output_dir results/cgan --num_samples 10 --noise_level 0.1
```

## Evaluation Metrics

The performance of the denoising CGAN is evaluated using:
- Mean Squared Error (MSE) between clean and denoised signals
- Improvement percentage compared to noisy signals
- Visual comparison of clean, noisy, and denoised HRRP signals

## Results

The CGAN demonstrates effective denoising capability, preserving important features of the HRRP signals while removing Gaussian noise. The conditioning on target identity and radial length information helps in maintaining target-specific characteristics in the denoised output.

Sample results are saved in the `results/cgan` directory, including visualizations of clean, noisy, and denoised HRRP signals along with quantitative metrics.

## License

This project is licensed under the MIT License - see the LICENSE.txt file for details.

## Acknowledgments

This implementation builds upon existing work in conditional GANs and radar signal processing techniques, adapting them for the specific task of HRRP signal denoising.