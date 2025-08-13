import os
import pandas as pd
import random
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from albumentations import Compose, Normalize, Resize
from albumentations import (
    RandomResizedCrop,
    HorizontalFlip,
    VerticalFlip,
    RandomBrightnessContrast,
    ColorJitter,
    GaussNoise,
    RandomRotate90,
    ShiftScaleRotate,
    CoarseDropout,
    RandomGamma,
    CLAHE,
    Blur,
    OneOf,
    RandomResizedCrop,
    HorizontalFlip,
    VerticalFlip,
    RandomRotate90,
    ShiftScaleRotate,
    OpticalDistortion,
    GridDistortion,
    ElasticTransform,
    RandomBrightnessContrast,
    ColorJitter,
    HueSaturationValue,
    RGBShift,
    GaussNoise,
    ISONoise,
    MotionBlur,
    Blur,
    CLAHE,
    CoarseDropout,
    ChannelShuffle,
    Normalize,
    OneOf,
    Resize,
)
from albumentations.pytorch import ToTensorV2
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from torchvision import models
from sklearn.model_selection import train_test_split
from logging import getLogger, DEBUG, FileHandler, Formatter, StreamHandler
import tqdm
import numpy as np
from PIL import Image
import time
import csv
from collections import Counter


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.

    Args:
        alpha (float): Weighting factor for rare class (default: 1.0)
        gamma (float): Focusing parameter (default: 2.0)
        reduction (str): Specifies the reduction to apply to the output
    """

    def __init__(self, alpha=1.0, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


def ensure_folder(folder):
    """
    Ensure a folder exists; if not, create it.
    """
    if not os.path.exists(folder):
        print(f"Folder '{folder}' does not exist. Creating...")
        os.makedirs(folder)


def seed_torch(seed=777):
    """
    Set seed for reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def initialize_csv_logger(file_path):
    """Initialize the CSV file with header."""
    header = [
        "epoch",
        "time",
        "val_loss",
        "val_accuracy",
        "train_loss",
        "train_accuracy",
    ]
    with open(file_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)


def log_epoch_to_csv(
    file_path, epoch, epoch_time, train_loss, train_accuracy, val_loss, val_accuracy
):
    """Log epoch summary to the CSV file."""
    with open(file_path, mode="a", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [epoch, epoch_time, val_loss, val_accuracy, train_loss, train_accuracy]
        )


def get_transforms(data):
    """
    Return augmentation transforms for the specified mode ('train' or 'valid').
    """
    width, height = 224 * 4, 224 * 4
    if data == "train":
        return Compose(
            [
                RandomResizedCrop(
                    (width, height), scale=(0.7, 1.0), ratio=(0.8, 1.2), p=1.0
                ),
                # Geometric variations
                OneOf(
                    [
                        HorizontalFlip(p=1.0),
                        VerticalFlip(p=1.0),
                        RandomRotate90(p=1.0),
                        ShiftScaleRotate(
                            shift_limit=0.15, scale_limit=0.15, rotate_limit=25, p=1.0
                        ),
                    ],
                    p=0.8,
                ),
                # Distortion-based augmentations (mimic lens effects)
                OneOf(
                    [
                        OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=1.0),
                        GridDistortion(num_steps=5, distort_limit=0.03, p=1.0),
                        ElasticTransform(alpha=50, sigma=5, alpha_affine=10, p=1.0),
                    ],
                    p=0.4,
                ),
                # Color and lighting variations
                OneOf(
                    [
                        ColorJitter(
                            brightness=0.3,
                            contrast=0.3,
                            saturation=0.3,
                            hue=0.15,
                            p=1.0,
                        ),
                        RandomBrightnessContrast(
                            brightness_limit=0.3, contrast_limit=0.3, p=1.0
                        ),
                        HueSaturationValue(
                            hue_shift_limit=15,
                            sat_shift_limit=20,
                            val_shift_limit=20,
                            p=1.0,
                        ),
                        RGBShift(
                            r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1.0
                        ),
                    ],
                    p=0.7,
                ),
                # Noise & blur
                OneOf(
                    [
                        GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                        ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
                        MotionBlur(blur_limit=3, p=1.0),
                        Blur(blur_limit=3, p=1.0),
                    ],
                    p=0.4,
                ),
                # Small-scale occlusion & channel mixing
                CoarseDropout(
                    max_holes=8, max_height=32, max_width=32, fill_value=0, p=0.3
                ),
                ChannelShuffle(p=0.1),
                # Contrast enhancement
                CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.2),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )
    elif data == "valid":
        return Compose(
            [
                Resize(height=height, width=width),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )
    else:
        raise ValueError(
            "Unknown data mode requested (only 'train' or 'valid' allowed)."
        )


class FungiDataset(Dataset):
    def __init__(self, df, path, transform=None):
        self.df = df
        self.transform = transform
        self.path = path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.df["filename_index"].values[idx]
        # Get label if it exists; otherwise return None
        label = self.df["taxonID_index"].values[idx]  # Get label
        if pd.isnull(label):
            label = -1  # Handle missing labels for the test dataset
        else:
            label = int(label)

        with Image.open(os.path.join(self.path, file_path)) as img:
            # Convert to RGB mode (handles grayscale images as well)
            image = img.convert("RGB")
        image = np.array(image)

        # Apply transformations if available
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image, label, file_path


def train_fungi_network(data_file, image_path, checkpoint_dir):
    """
    Train the network and save the best models based on validation accuracy and loss.
    Incorporates early stopping with a patience of 10 epochs.
    """
    # Ensure checkpoint directory exists
    ensure_folder(checkpoint_dir)

    # Set Logger
    csv_file_path = os.path.join(checkpoint_dir, "train.csv")
    initialize_csv_logger(csv_file_path)

    # Load metadata
    df = pd.read_csv(data_file)
    train_df = df[df["filename_index"].str.startswith("fungi_train")]
    train_df, val_df = train_test_split(train_df, test_size=0.015, random_state=42)
    print("Training size", len(train_df))
    print("Validation size", len(val_df))

    # Initialize DataLoaders
    train_dataset = FungiDataset(
        train_df, image_path, transform=get_transforms(data="train")
    )
    valid_dataset = FungiDataset(
        val_df, image_path, transform=get_transforms(data="valid")
    )
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=4)
    valid_loader = DataLoader(
        valid_dataset, batch_size=10, shuffle=False, num_workers=4
    )

    # Network Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.efficientnet_v2_l(pretrained=True)
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(
            model.classifier[1].in_features, len(train_df["taxonID_index"].unique())
        ),
    )
    model.to(device)

    # Define Optimization, Scheduler, and Criterion
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, "min", factor=0.9, patience=1, eps=1e-6)
    # criterion = FocalLoss(alpha=1.0, gamma=0.4, reduction="mean")
    criterion = nn.CrossEntropyLoss()

    # Early stopping setup
    patience = 10
    patience_counter = 0
    best_loss = np.inf
    best_accuracy = 0.0

    # Training Loop
    for epoch in range(100):  # Maximum epochs
        model.train()
        train_loss = 0.0
        total_correct_train = 0
        total_train_samples = 0

        # Start epoch timer
        epoch_start_time = time.time()

        # Training Loop
        for images, labels, _ in tqdm.tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Calculate train accuracy
            total_correct_train += (outputs.argmax(1) == labels).sum().item()
            total_train_samples += labels.size(0)

        # Calculate overall train accuracy and average loss
        train_accuracy = total_correct_train / total_train_samples
        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        total_correct_val = 0
        total_val_samples = 0

        # Validation Loop
        with torch.no_grad():
            for images, labels, _ in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()

                # Calculate validation accuracy
                total_correct_val += (outputs.argmax(1) == labels).sum().item()
                total_val_samples += labels.size(0)

        # Calculate overall validation accuracy and average loss
        val_accuracy = total_correct_val / total_val_samples
        avg_val_loss = val_loss / len(valid_loader)

        # Stop epoch timer and calculate elapsed time
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time

        # Print summary at the end of the epoch
        print(
            f"Epoch {epoch + 1} Summary: "
            f"Train Loss = {avg_train_loss:.4f}, Train Accuracy = {train_accuracy:.4f}, "
            f"Val Loss = {avg_val_loss:.4f}, Val Accuracy = {val_accuracy:.4f}, "
            f"Epoch Time = {epoch_time:.2f} seconds"
        )

        # Log epoch metrics to the CSV file
        log_epoch_to_csv(
            csv_file_path,
            epoch + 1,
            epoch_time,
            avg_train_loss,
            train_accuracy,
            avg_val_loss,
            val_accuracy,
        )

        # Save Models Based on Accuracy and Loss
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(
                model.state_dict(), os.path.join(checkpoint_dir, "best_accuracy.pth")
            )
            print(f"Epoch {epoch + 1}: Best accuracy updated to {best_accuracy:.4f}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(
                model.state_dict(), os.path.join(checkpoint_dir, "best_loss.pth")
            )
            print(f"Epoch {epoch + 1}: Best loss updated to {best_loss:.4f}")
            patience_counter = 0  # Reset patience counter
        else:
            patience_counter += 1

        # Early stopping condition
        if patience_counter >= patience:
            print(
                f"Early stopping triggered. No improvement in validation loss for {patience} epochs."
            )
            break


def evaluate_network_on_test_set(data_file, image_path, checkpoint_dir, session_name):
    """
    Evaluate network on the test set and save predictions to a CSV file.
    """
    # Ensure checkpoint directory exists
    ensure_folder(checkpoint_dir)

    # Model and Test Setup
    best_trained_model = os.path.join(checkpoint_dir, "best_accuracy.pth")
    output_csv_path = os.path.join(checkpoint_dir, "test_predictions.csv")

    df = pd.read_csv(data_file)
    test_df = df[df["filename_index"].str.startswith("fungi_test")]
    test_dataset = FungiDataset(
        test_df, image_path, transform=get_transforms(data="valid")
    )
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.efficientnet_v2_l(pretrained=True)
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.classifier[1].in_features, 183),  # Number of classes
    )
    model.load_state_dict(torch.load(best_trained_model))
    model.to(device)

    # Collect Predictions
    results = []
    model.eval()
    with torch.no_grad():
        for images, labels, filenames in tqdm.tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images).argmax(1).cpu().numpy()
            results.extend(
                zip(filenames, outputs)
            )  # Store filenames and predictions only

    # Save Results to CSV
    with open(output_csv_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([session_name])  # Write session name as the first line
        writer.writerows(results)  # Write filenames and predictions
    print(f"Results saved to {output_csv_path}")


def evaluate_network_on_test_set_with_tta(
    data_file, image_path, checkpoint_dir, session_name, tta_rounds=5
):
    """
    Evaluate network on the test set with Test Time Augmentation (TTA) and save predictions to a CSV file.
    Uses batch inference for faster processing.

    Args:
        data_file: Path to the metadata CSV file
        image_path: Path to the images directory
        checkpoint_dir: Directory containing the trained model checkpoint
        session_name: Name of the session for output file
        tta_rounds: Number of augmented versions to generate per image for averaging
    """
    # Ensure checkpoint directory exists
    ensure_folder(checkpoint_dir)

    # Model and Test Setup
    best_trained_model = os.path.join(checkpoint_dir, "best_accuracy.pth")
    output_csv_path = os.path.join(
        checkpoint_dir, f"test_predictions_tta_{tta_rounds}.csv"
    )

    df = pd.read_csv(data_file)
    test_df = df[df["filename_index"].str.startswith("fungi_test")]

    # Create datasets with different transforms
    test_dataset_clean = FungiDataset(
        test_df, image_path, transform=get_transforms(data="valid")
    )
    test_dataset_augmented = FungiDataset(
        test_df, image_path, transform=get_transforms(data="train")
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.efficientnet_v2_l(pretrained=True)
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.classifier[1].in_features, 183),  # Number of classes
    )
    model.load_state_dict(torch.load(best_trained_model))
    model.to(device)

    # Collect Predictions with TTA
    results = []
    tensor_results = []  # Store tensors and filenames
    model.eval()

    with torch.no_grad():
        # Process each image with TTA
        for idx in tqdm.tqdm(range(len(test_df)), desc="Evaluating with TTA"):
            # Get clean image
            clean_image, _, filename = test_dataset_clean[idx]

            # Generate augmented versions of the same image
            augmented_images = []
            for _ in range(tta_rounds):
                aug_image, _, _ = test_dataset_augmented[idx]
                augmented_images.append(aug_image)

            # Stack all images (clean + augmented) for batch inference
            all_images = torch.stack([clean_image] + augmented_images).to(device)

            # Batch inference - much faster than individual predictions
            all_outputs = model(all_images)
            # print(all_outputs.shape)
            # print(all_outputs)
            # print(all_outputs.argmax(1))

            # Average predictions across all augmentations
            averaged_output = torch.mean(all_outputs, dim=0, keepdim=True)

            # Get final prediction
            prediction = averaged_output.argmax(1).cpu().numpy()[0]
            results.append((filename, prediction))

            # Store tensor and filename for saving
            tensor_results.append(
                {
                    "filename": filename,
                    "averaged_output": averaged_output.cpu(),  # Move to CPU for saving
                }
            )

    # Save Results to CSV
    with open(output_csv_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([session_name])  # Write session name as the first line
        writer.writerows(results)  # Write filenames and predictions
    print(f"TTA Results saved to {output_csv_path}")

    # Save averaged output tensors
    tensors_output_path = os.path.join(
        checkpoint_dir, f"averaged_tensors_tta_{tta_rounds}.pt"
    )
    torch.save(tensor_results, tensors_output_path)
    print(f"Averaged output tensors saved to {tensors_output_path}")


if __name__ == "__main__":
    # Path to fungi images
    image_path = "/work3/monka/SummerSchool2025/FungiImages/"
    # Path to metadata file
    data_file = str("/work3/monka/SummerSchool2025/metadata.csv")

    # Session name: Change session name for every experiment!
    # Session name will be saved as the first line of the prediction file
    session = "EfficientNet_V2L_CrossEntropy_InputShapeIncreased"

    # Folder for results of this experiment based on session name:
    checkpoint_dir = os.path.join(f"/work3/monka/SummerSchool2025/results/{session}/")

    train_fungi_network(data_file, image_path, checkpoint_dir)
    evaluate_network_on_test_set(data_file, image_path, checkpoint_dir, session)
    evaluate_network_on_test_set_with_tta(
        data_file, image_path, checkpoint_dir, session, tta_rounds=10
    )
    evaluate_network_on_test_set_with_tta(
        data_file, image_path, checkpoint_dir, session, tta_rounds=64
    )
