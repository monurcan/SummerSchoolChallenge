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


def load_fold_indices(checkpoint_dir):
    """
    Load previously saved fold indices.

    Returns:
        dict: Dictionary containing all saved indices and file lists
    """
    fold_indices_dir = os.path.join(checkpoint_dir, "fold_indices")

    if not os.path.exists(fold_indices_dir):
        raise FileNotFoundError(f"Fold indices directory not found: {fold_indices_dir}")

    indices_data = {}

    # Load subset indices
    indices_data["subset_1_indices"] = np.load(
        os.path.join(fold_indices_dir, "subset_1_indices.npy")
    )
    indices_data["subset_2_indices"] = np.load(
        os.path.join(fold_indices_dir, "subset_2_indices.npy")
    )
    indices_data["subset_3_indices"] = np.load(
        os.path.join(fold_indices_dir, "subset_3_indices.npy")
    )
    indices_data["all_shuffled_indices"] = np.load(
        os.path.join(fold_indices_dir, "all_shuffled_indices.npy")
    )

    # Load fold-specific indices
    for fold in range(1, 4):
        indices_data[f"fold_{fold}_train_indices"] = np.load(
            os.path.join(fold_indices_dir, f"fold_{fold}_train_indices.npy")
        )
        indices_data[f"fold_{fold}_val_indices"] = np.load(
            os.path.join(fold_indices_dir, f"fold_{fold}_val_indices.npy")
        )

    print(f"Loaded fold indices from: {fold_indices_dir}")
    return indices_data


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
    width, height = 224, 224
    if data == "train":
        return Compose(
            [
                RandomResizedCrop((width, height), scale=(0.8, 1.0)),
                OneOf(
                    [
                        HorizontalFlip(p=1.0),
                        VerticalFlip(p=1.0),
                        RandomRotate90(p=1.0),
                    ],
                    p=0.7,
                ),
                OneOf(
                    [
                        ColorJitter(
                            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0
                        ),
                        RandomBrightnessContrast(p=1.0),
                        RandomGamma(gamma_limit=(80, 120), p=1.0),
                    ],
                    p=0.5,
                ),
                OneOf(
                    [
                        GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                        Blur(blur_limit=3, p=1.0),
                    ],
                    p=0.3,
                ),
                ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.3
                ),
                CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.3),
                CLAHE(clip_limit=2.0, p=0.2),
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


def train_fungi_network(data_file, image_path, checkpoint_dir, fold_num=None):
    """
    Train the network and save the best models based on validation accuracy and loss.
    Incorporates early stopping with a patience of 10 epochs.
    If fold_num is provided, uses 3-fold cross-validation.
    """
    # Ensure checkpoint directory exists
    ensure_folder(checkpoint_dir)

    # Set Logger
    if fold_num is not None:
        csv_file_path = os.path.join(checkpoint_dir, f"train_validation{fold_num}.csv")
    else:
        csv_file_path = os.path.join(checkpoint_dir, "train.csv")
    initialize_csv_logger(csv_file_path)

    # Load metadata
    df = pd.read_csv(data_file)
    train_df = df[df["filename_index"].str.startswith("fungi_train")]

    if fold_num is not None:
        # 3-fold cross-validation
        # Split the data into 3 roughly equal subsets
        train_df = train_df.reset_index(drop=True)
        n_samples = len(train_df)
        indices = np.arange(n_samples)
        np.random.seed(42)  # For reproducibility
        np.random.shuffle(indices)

        # Create 3 folds
        fold_size = n_samples // 3
        subset_1_indices = indices[:fold_size]
        subset_2_indices = indices[fold_size : 2 * fold_size]
        subset_3_indices = indices[2 * fold_size :]

        # Save fold indices to files for reproducibility
        fold_indices_dir = os.path.join(checkpoint_dir, "fold_indices")
        ensure_folder(fold_indices_dir)

        # Save all subset indices
        np.save(
            os.path.join(fold_indices_dir, "subset_1_indices.npy"), subset_1_indices
        )
        np.save(
            os.path.join(fold_indices_dir, "subset_2_indices.npy"), subset_2_indices
        )
        np.save(
            os.path.join(fold_indices_dir, "subset_3_indices.npy"), subset_3_indices
        )
        np.save(os.path.join(fold_indices_dir, "all_shuffled_indices.npy"), indices)

        # Also save original filenames for each subset for easy reference
        subset_1_files = train_df.iloc[subset_1_indices]["filename_index"].tolist()
        subset_2_files = train_df.iloc[subset_2_indices]["filename_index"].tolist()
        subset_3_files = train_df.iloc[subset_3_indices]["filename_index"].tolist()

        with open(os.path.join(fold_indices_dir, "subset_1_files.txt"), "w") as f:
            f.write("\n".join(subset_1_files))
        with open(os.path.join(fold_indices_dir, "subset_2_files.txt"), "w") as f:
            f.write("\n".join(subset_2_files))
        with open(os.path.join(fold_indices_dir, "subset_3_files.txt"), "w") as f:
            f.write("\n".join(subset_3_files))

        # Select validation and training sets based on fold_num
        if fold_num == 1:
            val_indices = subset_1_indices
            train_indices = np.concatenate([subset_2_indices, subset_3_indices])
            print(
                f"Fold {fold_num}: Using subset_1 for validation, subset_2 & subset_3 for training"
            )
        elif fold_num == 2:
            val_indices = subset_2_indices
            train_indices = np.concatenate([subset_1_indices, subset_3_indices])
            print(
                f"Fold {fold_num}: Using subset_2 for validation, subset_1 & subset_3 for training"
            )
        elif fold_num == 3:
            val_indices = subset_3_indices
            train_indices = np.concatenate([subset_1_indices, subset_2_indices])
            print(
                f"Fold {fold_num}: Using subset_3 for validation, subset_1 & subset_2 for training"
            )
        else:
            raise ValueError("fold_num must be 1, 2, or 3")

        # Save current fold's train and validation indices
        np.save(
            os.path.join(fold_indices_dir, f"fold_{fold_num}_train_indices.npy"),
            train_indices,
        )
        np.save(
            os.path.join(fold_indices_dir, f"fold_{fold_num}_val_indices.npy"),
            val_indices,
        )

        # Save current fold's train and validation filenames
        train_files = train_df.iloc[train_indices]["filename_index"].tolist()
        val_files = train_df.iloc[val_indices]["filename_index"].tolist()

        with open(
            os.path.join(fold_indices_dir, f"fold_{fold_num}_train_files.txt"), "w"
        ) as f:
            f.write("\n".join(train_files))
        with open(
            os.path.join(fold_indices_dir, f"fold_{fold_num}_val_files.txt"), "w"
        ) as f:
            f.write("\n".join(val_files))

        train_df_fold = train_df.iloc[train_indices].reset_index(drop=True)
        val_df = train_df.iloc[val_indices].reset_index(drop=True)
        train_df = train_df_fold

        print(f"Fold indices saved to: {fold_indices_dir}")
        print(
            f"- Subset sizes: {len(subset_1_indices)}, {len(subset_2_indices)}, {len(subset_3_indices)}"
        )
        print(
            f"- Current fold train size: {len(train_indices)}, val size: {len(val_indices)}"
        )
    else:
        # Original single split
        train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

    print("Training size", len(train_df))
    print("Validation size", len(val_df))

    # Initialize DataLoaders
    train_dataset = FungiDataset(
        train_df, image_path, transform=get_transforms(data="train")
    )
    valid_dataset = FungiDataset(
        val_df, image_path, transform=get_transforms(data="valid")
    )
    train_loader = DataLoader(
        train_dataset, batch_size=256, shuffle=True, num_workers=4
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=256, shuffle=False, num_workers=4
    )

    # Network Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.efficientnet_b2(pretrained=True)
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
    criterion = FocalLoss(alpha=1.0, gamma=0.4, reduction="mean")

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
        for images, labels, _ in train_loader:
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
        if fold_num is not None:
            # Save with fold-specific names
            best_accuracy_filename = f"best_accuracy_validation{fold_num}.pth"
            best_loss_filename = f"best_loss_validation{fold_num}.pth"
        else:
            # Original filenames
            best_accuracy_filename = "best_accuracy.pth"
            best_loss_filename = "best_loss.pth"

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(
                model.state_dict(), os.path.join(checkpoint_dir, best_accuracy_filename)
            )
            print(f"Epoch {epoch + 1}: Best accuracy updated to {best_accuracy:.4f}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(
                model.state_dict(), os.path.join(checkpoint_dir, best_loss_filename)
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


def evaluate_network_on_test_set(
    data_file, image_path, checkpoint_dir, session_name, fold_num=None
):
    """
    Evaluate network on the test set and save predictions to a CSV file.
    """
    # Ensure checkpoint directory exists
    ensure_folder(checkpoint_dir)

    # Model and Test Setup
    if fold_num is not None:
        best_trained_model = os.path.join(
            checkpoint_dir, f"best_accuracy_validation{fold_num}.pth"
        )
        output_csv_path = os.path.join(
            checkpoint_dir, f"test_predictions_validation{fold_num}.csv"
        )
    else:
        best_trained_model = os.path.join(checkpoint_dir, "best_accuracy.pth")
        output_csv_path = os.path.join(checkpoint_dir, "test_predictions.csv")

    df = pd.read_csv(data_file)
    test_df = df[df["filename_index"].str.startswith("fungi_test")]
    test_dataset = FungiDataset(
        test_df, image_path, transform=get_transforms(data="valid")
    )
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.efficientnet_b2(pretrained=True)
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


if __name__ == "__main__":
    # Path to fungi images
    image_path = "/work3/monka/SummerSchool2025/FungiImages/"
    # Path to metadata file
    data_file = str("/work3/monka/SummerSchool2025/metadata.csv")

    # Session name: Change session name for every experiment!
    # Session name will be saved as the first line of the prediction file
    session = "EfficientNetB2_FocalLossLess_3FoldCV"

    # Folder for results of this experiment based on session name:
    checkpoint_dir = os.path.join(f"/work3/monka/SummerSchool2025/results/{session}/")

    # Set seed for reproducibility
    seed_torch(777)

    # Run 3-fold cross-validation
    for fold in range(1, 4):
        print(f"\n{'=' * 50}")
        print(f"Starting training for fold {fold}")
        print(f"{'=' * 50}")

        train_fungi_network(data_file, image_path, checkpoint_dir, fold_num=fold)
        evaluate_network_on_test_set(
            data_file,
            image_path,
            checkpoint_dir,
            f"{session}_fold{fold}",
            fold_num=fold,
        )

        print(f"Completed fold {fold}")

    print(f"\n{'=' * 50}")
    print("3-fold cross-validation completed!")
    print("Models saved with names:")
    print("- best_accuracy_validation1.pth")
    print("- best_accuracy_validation2.pth")
    print("- best_accuracy_validation3.pth")
    print("- best_loss_validation1.pth")
    print("- best_loss_validation2.pth")
    print("- best_loss_validation3.pth")
    print("\nFold indices saved for future use:")
    print("- fold_indices/subset_1_indices.npy")
    print("- fold_indices/subset_2_indices.npy")
    print("- fold_indices/subset_3_indices.npy")
    print("- fold_indices/fold_X_train_indices.npy")
    print("- fold_indices/fold_X_val_indices.npy")
    print("- fold_indices/fold_X_train_files.txt")
    print("- fold_indices/fold_X_val_files.txt")
    print("\nTo load indices later, use: load_fold_indices(checkpoint_dir)")
    print(f"{'=' * 50}")
