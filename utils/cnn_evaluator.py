# cnn_evaluator.py - CNN model for evaluating denoising performance
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm


class HRRPCNN(nn.Module):
    """
    1D CNN model for HRRP classification based on the paper.
    Architecture:
    - Two convolutional blocks (conv + ReLU, kernel size 3, stride 2)
    - Two max-pooling layers (kernel size 3, stride 2)
    - Two FC layers with ReLU activation
    - Dropout layer (rate = 0.5)
    - Final FC layer with softmax for classification
    """

    def __init__(self, input_dim=500, num_classes=3):
        super(HRRPCNN, self).__init__()

        # First convolutional block
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, stride=2, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Second convolutional block
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Fully connected layers - we'll determine the input size in the forward pass
        self.fc1 = None  # Will be initialized in the first forward pass
        self.fc2 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc3 = nn.Linear(128, num_classes)

        self.initialized = False

    def _initialize_fc_layer(self, x_size):
        """Initialize the first fully connected layer based on the actual flattened size"""
        # Get device from existing parameters instead of x_size (which is an integer)
        self.fc1 = nn.Linear(x_size, 256).to(self.conv1.weight.device)
        self.initialized = True
        print(f"Initialized FC1 layer with input size: {x_size}")

    def forward(self, x):
        # Ensure input has the right shape [batch_size, 1, input_dim]
        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        # First convolutional block
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        # Second convolutional block
        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        # Flatten for fully connected layers
        batch_size = x.size(0)
        x = x.view(batch_size, -1)

        # Initialize FC1 if this is the first forward pass
        if not self.initialized:
            self._initialize_fc_layer(x.size(1))

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x

    def predict(self, x):
        """Get the predicted class"""
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)


def train_cnn(model, train_loader, val_loader=None, num_epochs=50, lr=0.001,
              device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
              save_path=None):
    """
    Train the CNN model for HRRP classification

    Args:
        model (HRRPCNN): The CNN model to train
        train_loader (DataLoader): DataLoader for training data
        val_loader (DataLoader): DataLoader for validation data (optional)
        num_epochs (int): Number of training epochs
        lr (float): Learning rate
        device (torch.device): Device to train on
        save_path (str): Path to save the best model

    Returns:
        dict: Training history (loss and accuracy)
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    # Initialize variables to track best model
    best_val_acc = 0.0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    # First pass a batch through the model to initialize FC1 before creating optimizer
    for batch in train_loader:
        # Get sample batch to initialize model
        inputs = batch[0]
        inputs = inputs.float().to(device)
        # Forward pass to initialize fc1
        model(inputs)
        # Break after one batch
        break

    # Now create optimizer after all parameters are initialized
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for batch in pbar:
            # HRRP dataset returns (data, radial_length, identity_label)
            # We only need data and identity_label for the CNN
            inputs = batch[0]
            labels = batch[2]

            inputs = inputs.float().to(device)
            labels = labels.long().to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            # Track statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'acc': 100 * correct / total
            })

        # Calculate epoch statistics
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)

        # Validate if validation data is provided
        if val_loader is not None:
            val_loss, val_acc = evaluate_cnn(model, val_loader, criterion, device)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            # Update learning rate based on validation loss
            scheduler.step(val_loss)

            # Save best model
            if val_acc > best_val_acc and save_path is not None:
                best_val_acc = val_acc
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(model.state_dict(), save_path)
                print(f"Saved best model with validation accuracy: {val_acc:.2f}%")

            # Print epoch summary
            print(f"Epoch {epoch + 1}/{num_epochs} - "
                  f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        else:
            print(f"Epoch {epoch + 1}/{num_epochs} - "
                  f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%")

    # If no validation data, save the final model
    if val_loader is None and save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"Saved final model with training accuracy: {epoch_acc:.2f}%")

    return history


def evaluate_cnn(model, data_loader, criterion=None,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """
    Evaluate the CNN model on the given data

    Args:
        model (HRRPCNN): The CNN model to evaluate
        data_loader (DataLoader): DataLoader for evaluation data
        criterion (torch.nn.Module): Loss function (optional)
        device (torch.device): Device to evaluate on

    Returns:
        tuple: (loss, accuracy)
    """
    model = model.to(device)
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in data_loader:
            # HRRP dataset returns (data, radial_length, identity_label)
            # We only need data and identity_label for the CNN
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                inputs = batch[0]
                labels = batch[2]
            elif isinstance(batch, (list, tuple)) and len(batch) == 2:
                # For cases when we're using TensorDataset (clean/noisy/denoised evaluation)
                inputs, labels = batch
            else:
                raise ValueError("Unexpected batch format")

            inputs = inputs.float().to(device)
            labels = labels.long().to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Track statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate statistics
    avg_loss = running_loss / len(data_loader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy


def create_cnn_dataloaders(signals, labels, batch_size=32, train_split=0.8, val_split=0.1, shuffle=True):
    """
    Create DataLoaders for training, validation, and testing from signals and labels

    Args:
        signals (torch.Tensor or numpy.ndarray): Signal data [num_samples, signal_length]
        labels (torch.Tensor or numpy.ndarray): Class labels [num_samples]
        batch_size (int): Batch size
        train_split (float): Proportion of data to use for training
        val_split (float): Proportion of data to use for validation
        shuffle (bool): Whether to shuffle the data

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Convert to PyTorch tensors if needed
    if isinstance(signals, np.ndarray):
        signals = torch.from_numpy(signals).float()
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels).long()

    # Calculate splits
    num_samples = signals.shape[0]
    train_size = int(train_split * num_samples)
    val_size = int(val_split * num_samples)
    test_size = num_samples - train_size - val_size

    # Create indices for splitting
    indices = torch.randperm(num_samples) if shuffle else torch.arange(num_samples)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    # Create TensorDatasets
    train_dataset = TensorDataset(signals[train_indices], labels[train_indices])
    val_dataset = TensorDataset(signals[val_indices], labels[val_indices]) if val_size > 0 else None
    test_dataset = TensorDataset(signals[test_indices], labels[test_indices]) if test_size > 0 else None

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size) if val_dataset else None
    test_loader = DataLoader(test_dataset, batch_size=batch_size) if test_dataset else None

    return train_loader, val_loader, test_loader


def evaluate_denoising_with_cnn(clean_data, noisy_data, denoised_data, cnn_model, device=None):
    """
    Evaluate denoising performance using CNN classifier accuracy

    Args:
        clean_data (torch.Tensor): Clean HRRP data [num_samples, signal_length]
        noisy_data (torch.Tensor): Noisy HRRP data [num_samples, signal_length]
        denoised_data (torch.Tensor): Denoised HRRP data [num_samples, signal_length]
        cnn_model (HRRPCNN): Trained CNN model for evaluation
        device (torch.device): Device to evaluate on

    Returns:
        dict: Evaluation results including CNN accuracy on clean, noisy, and denoised data
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cnn_model = cnn_model.to(device)
    cnn_model.eval()

    # Get the labels from clean data (using the CNN's prediction on clean data as ground truth)
    with torch.no_grad():
        clean_logits = cnn_model(clean_data.to(device))
        clean_preds = torch.argmax(clean_logits, dim=1)

    # Evaluate on noisy data
    with torch.no_grad():
        noisy_logits = cnn_model(noisy_data.to(device))
        noisy_preds = torch.argmax(noisy_logits, dim=1)
        noisy_accuracy = 100 * (noisy_preds == clean_preds).sum().item() / len(clean_preds)

    # Evaluate on denoised data
    with torch.no_grad():
        denoised_logits = cnn_model(denoised_data.to(device))
        denoised_preds = torch.argmax(denoised_logits, dim=1)
        denoised_accuracy = 100 * (denoised_preds == clean_preds).sum().item() / len(clean_preds)

    # Clean data accuracy (should be 100% as we're using its predictions as ground truth)
    clean_accuracy = 100.0

    return {
        'clean_accuracy': clean_accuracy,
        'noisy_accuracy': noisy_accuracy,
        'denoised_accuracy': denoised_accuracy,
        'accuracy_improvement': denoised_accuracy - noisy_accuracy
    }


def plot_training_history(history, save_path=None):
    """
    Plot training history

    Args:
        history (dict): Training history from train_cnn
        save_path (str): Path to save the plot (optional)

    Returns:
        matplotlib.figure.Figure: Figure object
    """
    plt.figure(figsize=(12, 5))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    if 'val_loss' in history and history['val_loss']:
        plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    if 'val_acc' in history and history['val_acc']:
        plt.plot(history['val_acc'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return plt.gcf()