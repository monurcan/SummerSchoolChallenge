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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from torchvision import models
from sklearn.model_selection import train_test_split
import tqdm
import numpy as np
from PIL import Image
import time
import csv
import pickle


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


class MetadataProcessor:
    """
    Processes metadata features for the fungi dataset.
    Handles categorical and numerical features similar to XGBoost preprocessing.
    """

    def __init__(self):
        self.categorical_mappings = {}
        self.fitted = False

    def fit(self, df):
        """Fit the processor on training data."""
        # Process categorical features
        for col in ["Habitat", "Substrate"]:
            if col in df.columns:
                # Get unique categories, fill NaN with 'unknown'
                categories = df[col].fillna("unknown").astype(str).unique()
                # Sort for consistency
                categories = sorted(categories)
                self.categorical_mappings[col] = {
                    cat: idx for idx, cat in enumerate(categories)
                }

        self.fitted = True
        return self

    def transform(self, df):
        """Transform dataframe to processed features."""
        if not self.fitted:
            raise ValueError("Processor must be fitted before transform")

        processed_features = {}

        # Categorical features - convert to indices
        for col in ["Habitat", "Substrate"]:
            if col in df.columns:
                # Fill NaN and convert to string
                values = df[col].fillna("unknown").astype(str)
                # Map to indices, use 0 for unknown categories
                indices = []
                for val in values:
                    if val in self.categorical_mappings[col]:
                        indices.append(self.categorical_mappings[col][val])
                    else:
                        indices.append(0)  # Default to first category for unknown
                processed_features[col] = torch.tensor(indices, dtype=torch.long)
            else:
                # Create dummy feature if column doesn't exist
                processed_features[col] = torch.zeros(len(df), dtype=torch.long)

        # Numerical features
        for col in ["Latitude", "Longitude"]:
            if col in df.columns:
                values = df[col].fillna(0.0).astype(float).values
                processed_features[col] = torch.tensor(values, dtype=torch.float32)
            else:
                processed_features[col] = torch.zeros(len(df), dtype=torch.float32)

        # Temporal features from eventDate
        if "eventDate" in df.columns:
            dates = pd.to_datetime(df["eventDate"], errors="coerce")

            # Month (1-12, 0 for NaN)
            months = dates.dt.month.fillna(0).astype(int).values
            processed_features["month"] = torch.tensor(months, dtype=torch.long)

            # Year (normalized to 2000-2030 range, 0 for NaN)
            years = dates.dt.year.fillna(2010).astype(int).values
            # Normalize years to reasonable range
            years = (years - 2000) / 30.0  # Normalize to roughly [-0.67, 1.0] range
            processed_features["year"] = torch.tensor(years, dtype=torch.float32)

            # Season (0-3 for Winter, Spring, Summer, Autumn)
            seasons = months.copy()
            season_mapping = {
                12: 0,
                1: 0,
                2: 0,  # Winter
                3: 1,
                4: 1,
                5: 1,  # Spring
                6: 2,
                7: 2,
                8: 2,  # Summer
                9: 3,
                10: 3,
                11: 3,  # Autumn
                0: 0,
            }  # Default for missing
            seasons = (
                pd.Series(seasons).map(season_mapping).fillna(0).astype(int).values
            )
            processed_features["season"] = torch.tensor(seasons, dtype=torch.long)
        else:
            # Default temporal features
            processed_features["month"] = torch.zeros(len(df), dtype=torch.long)
            processed_features["year"] = torch.zeros(len(df), dtype=torch.float32)
            processed_features["season"] = torch.zeros(len(df), dtype=torch.long)

        return processed_features

    def fit_transform(self, df):
        """Fit and transform in one step."""
        return self.fit(df).transform(df)

    def get_feature_dims(self):
        """Get the dimensions for embedding layers."""
        if not self.fitted:
            raise ValueError("Processor must be fitted first")

        dims = {}
        # Categorical feature dimensions (number of categories)
        dims["Habitat"] = len(self.categorical_mappings.get("Habitat", {}))
        dims["Substrate"] = len(self.categorical_mappings.get("Substrate", {}))
        dims["month"] = 13  # 0-12 (0 for missing, 1-12 for actual months)
        dims["season"] = 4  # 0-3 for seasons

        # Numerical features (dimension 1 each)
        dims["Latitude"] = 1
        dims["Longitude"] = 1
        dims["year"] = 1

        return dims


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
    width, height = 224 * 2, 224 * 2
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
                        GaussNoise(var_limit=(10, 50), p=1.0),
                        Blur(blur_limit=3, p=1.0),
                    ],
                    p=0.3,
                ),
                ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.3
                ),
                CoarseDropout(
                    num_holes_range=(1, 8),
                    hole_height_range=(8, 16),
                    hole_width_range=(8, 16),
                    p=0.3,
                ),
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
    def __init__(self, df, path, transform=None, metadata_processor=None):
        self.df = df
        self.transform = transform
        self.path = path
        self.metadata_processor = metadata_processor

        # Pre-process metadata for efficiency
        if metadata_processor is not None:
            self.metadata_features = metadata_processor.transform(df)
        else:
            self.metadata_features = None

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

        # Get metadata features for this sample
        metadata = {}
        if self.metadata_features is not None:
            for key, values in self.metadata_features.items():
                metadata[key] = values[idx]

        return image, label, file_path, metadata


class MultimodalFungiModel(nn.Module):
    """
    Multimodal model that combines EfficientNet image features with metadata features.
    """

    def __init__(self, num_classes, metadata_dims, pretrained_model_path=None):
        super(MultimodalFungiModel, self).__init__()

        # Load pre-trained EfficientNet
        self.backbone = models.efficientnet_v2_l(pretrained=True)

        # Get the feature dimension from the backbone
        backbone_feature_dim = self.backbone.classifier[1].in_features

        # Remove the original classifier
        self.backbone.classifier = nn.Identity()

        # Load pre-trained weights if provided
        if pretrained_model_path and os.path.exists(pretrained_model_path):
            print(f"Loading pre-trained image model from: {pretrained_model_path}")
            # Load the state dict from the image-only model
            pretrained_state = torch.load(pretrained_model_path, map_location="cpu")

            # Create a temporary model to load the pretrained weights
            temp_model = models.efficientnet_v2_l(pretrained=True)
            temp_model.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(temp_model.classifier[1].in_features, num_classes),
            )
            temp_model.load_state_dict(pretrained_state)

            # Copy the backbone weights (everything except classifier)
            backbone_dict = {}
            for name, param in temp_model.named_parameters():
                if not name.startswith("classifier"):
                    backbone_dict[name] = param

            self.backbone.load_state_dict(backbone_dict, strict=False)
            print("Pre-trained backbone weights loaded successfully!")

        # Metadata embedding layers
        self.habitat_embedding = nn.Embedding(
            metadata_dims["Habitat"] + 1, 16
        )  # +1 for padding
        self.substrate_embedding = nn.Embedding(metadata_dims["Substrate"] + 1, 16)
        self.month_embedding = nn.Embedding(metadata_dims["month"] + 1, 8)
        self.season_embedding = nn.Embedding(metadata_dims["season"] + 1, 4)

        # Calculate total metadata feature dimension
        metadata_feature_dim = (
            16
            + 16
            + 8
            + 4  # Embeddings: habitat + substrate + month + season
            + 2
            + 1  # Numerical: lat + lon + year
        )

        # Combined feature dimension
        combined_dim = backbone_feature_dim + metadata_feature_dim

        # Enhanced classifier with more layers to handle multimodal fusion
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(combined_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

        # Initialize embeddings
        self._initialize_embeddings()

    def _initialize_embeddings(self):
        """Initialize embedding layers with appropriate values."""
        nn.init.xavier_uniform_(self.habitat_embedding.weight)
        nn.init.xavier_uniform_(self.substrate_embedding.weight)
        nn.init.xavier_uniform_(self.month_embedding.weight)
        nn.init.xavier_uniform_(self.season_embedding.weight)

    def forward(self, images, metadata):
        # Extract image features
        image_features = self.backbone(images)  # [batch_size, backbone_feature_dim]

        # Process metadata features
        habitat_emb = self.habitat_embedding(metadata["Habitat"])
        substrate_emb = self.substrate_embedding(metadata["Substrate"])
        month_emb = self.month_embedding(metadata["month"])
        season_emb = self.season_embedding(metadata["season"])

        # Numerical features (already float tensors)
        lat = metadata["Latitude"].unsqueeze(1)
        lon = metadata["Longitude"].unsqueeze(1)
        year = metadata["year"].unsqueeze(1)

        # Concatenate all metadata features
        metadata_features = torch.cat(
            [habitat_emb, substrate_emb, month_emb, season_emb, lat, lon, year], dim=1
        )

        # Combine image and metadata features
        combined_features = torch.cat([image_features, metadata_features], dim=1)

        # Final classification
        output = self.classifier(combined_features)

        return output


def collate_fn(batch):
    """Custom collate function to handle metadata dictionaries."""
    images, labels, filenames, metadata_list = zip(*batch)

    # Stack images and labels
    images = torch.stack(images)
    labels = torch.tensor(labels)

    # Combine metadata dictionaries
    metadata = {}
    for key in metadata_list[0].keys():
        metadata[key] = torch.stack([m[key] for m in metadata_list])

    return images, labels, filenames, metadata


def train_fungi_network(
    data_file, image_path, checkpoint_dir, pretrained_model_path=None
):
    """
    Train the multimodal network with image and metadata features.
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

    # Initialize metadata processor
    print("Processing metadata...")
    metadata_processor = MetadataProcessor()
    metadata_processor.fit(train_df)
    metadata_dims = metadata_processor.get_feature_dims()

    print("Metadata dimensions:")
    for key, dim in metadata_dims.items():
        print(f"  {key}: {dim}")

    # Save metadata processor
    processor_path = os.path.join(checkpoint_dir, "metadata_processor.pkl")
    with open(processor_path, "wb") as f:
        pickle.dump(metadata_processor, f)
    print(f"Metadata processor saved to: {processor_path}")

    # Initialize DataLoaders with metadata processor
    train_dataset = FungiDataset(
        train_df,
        image_path,
        transform=get_transforms(data="train"),
        metadata_processor=metadata_processor,
    )
    valid_dataset = FungiDataset(
        val_df,
        image_path,
        transform=get_transforms(data="valid"),
        metadata_processor=metadata_processor,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=4, collate_fn=collate_fn
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
    )

    # Network Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalFungiModel(
        num_classes=len(train_df["taxonID_index"].unique()),
        metadata_dims=metadata_dims,
        pretrained_model_path=pretrained_model_path,
    )
    model.to(device)

    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Define Optimization and Criterion
    optimizer = AdamW(
        model.parameters(), lr=1e-4, weight_decay=1e-5
    )  # Lower LR for fine-tuning
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
        for images, labels, _, metadata in tqdm.tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Move metadata to device
            metadata = {k: v.to(device) for k, v in metadata.items()}

            optimizer.zero_grad()
            outputs = model(images, metadata)
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
            for images, labels, _, metadata in valid_loader:
                images, labels = images.to(device), labels.to(device)
                metadata = {k: v.to(device) for k, v in metadata.items()}

                outputs = model(images, metadata)
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
    Evaluate multimodal network on the test set and save predictions to a CSV file.
    """
    # Ensure checkpoint directory exists
    ensure_folder(checkpoint_dir)

    # Model and Test Setup
    best_trained_model = os.path.join(checkpoint_dir, "best_accuracy.pth")
    processor_path = os.path.join(checkpoint_dir, "metadata_processor.pkl")
    output_csv_path = os.path.join(checkpoint_dir, "test_predictions.csv")
    output_probs_path = os.path.join(checkpoint_dir, "test_probabilities.csv")

    # Load metadata processor
    with open(processor_path, "rb") as f:
        metadata_processor = pickle.load(f)
    metadata_dims = metadata_processor.get_feature_dims()

    df = pd.read_csv(data_file)
    test_df = df[df["filename_index"].str.startswith(("fungi_test", "fungi_final"))]
    test_dataset = FungiDataset(
        test_df,
        image_path,
        transform=get_transforms(data="valid"),
        metadata_processor=metadata_processor,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=4, collate_fn=collate_fn
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalFungiModel(num_classes=183, metadata_dims=metadata_dims)
    model.load_state_dict(torch.load(best_trained_model))
    model.to(device)

    # Collect Predictions and Probabilities
    results = []
    prob_results = []
    model.eval()
    with torch.no_grad():
        for images, labels, filenames, metadata in tqdm.tqdm(
            test_loader, desc="Evaluating"
        ):
            images = images.to(device)
            metadata = {k: v.to(device) for k, v in metadata.items()}

            outputs = model(images, metadata)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
            predictions = outputs.argmax(1).cpu().numpy()

            # Store predictions
            results.extend(zip(filenames, predictions))

            # Store probabilities with filenames
            for filename, prob_row in zip(filenames, probabilities):
                prob_results.append([filename] + prob_row.tolist())

    # Save Predictions to CSV
    with open(output_csv_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([session_name])  # Write session name as the first line
        writer.writerows(results)  # Write filenames and predictions
    print(f"Results saved to {output_csv_path}")

    # Save Probabilities to CSV
    with open(output_probs_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        # Write header with session name and class indices
        header = [session_name] + [f"class_{i}" for i in range(183)]
        writer.writerow(header)
        writer.writerows(prob_results)  # Write filenames and probabilities
    print(f"Probabilities saved to {output_probs_path}")


def evaluate_network_on_test_set_with_tta(
    data_file, image_path, checkpoint_dir, session_name, tta_rounds=5
):
    """
    Evaluate multimodal network on the test set with Test Time Augmentation (TTA).
    """
    # Ensure checkpoint directory exists
    ensure_folder(checkpoint_dir)

    # Model and Test Setup
    best_trained_model = os.path.join(checkpoint_dir, "best_accuracy.pth")
    processor_path = os.path.join(checkpoint_dir, "metadata_processor.pkl")
    output_csv_path = os.path.join(
        checkpoint_dir, f"test_predictions_tta_{tta_rounds}.csv"
    )
    output_probs_path = os.path.join(
        checkpoint_dir, f"test_probabilities_tta_{tta_rounds}.csv"
    )

    # Load metadata processor
    with open(processor_path, "rb") as f:
        metadata_processor = pickle.load(f)
    metadata_dims = metadata_processor.get_feature_dims()

    df = pd.read_csv(data_file)
    test_df = df[df["filename_index"].str.startswith(("fungi_test", "fungi_final"))]

    # Create datasets with different transforms
    test_dataset_clean = FungiDataset(
        test_df, image_path, metadata_processor=metadata_processor
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalFungiModel(num_classes=183, metadata_dims=metadata_dims)
    model.load_state_dict(torch.load(best_trained_model))
    model.to(device)

    # Collect Predictions and Probabilities with TTA
    results = []
    prob_results = []
    model.eval()

    # Batch processing parameters
    batch_size = 8  # Number of original images to process together
    total_samples = len(test_df)

    with torch.no_grad():
        # Process images in batches for efficiency
        for start_idx in tqdm.tqdm(
            range(0, total_samples, batch_size), desc="Evaluating with TTA"
        ):
            end_idx = min(start_idx + batch_size, total_samples)
            current_batch_size = end_idx - start_idx

            # Collect all images and augmentations for this batch
            batch_images = []
            batch_filenames = []
            batch_metadata_list = []

            for idx in range(start_idx, end_idx):
                # Get clean image and metadata
                clean_image, _, filename, metadata = test_dataset_clean[idx]
                batch_filenames.append(filename)

                # Add clean image and metadata
                batch_images.append(
                    get_transforms(data="valid")(image=clean_image)["image"]
                )
                batch_metadata_list.append(metadata)

                # Generate augmented versions of the same image (metadata stays the same)
                for _ in range(tta_rounds):
                    batch_images.append(
                        get_transforms(data="train")(image=clean_image)["image"]
                    )
                    batch_metadata_list.append(
                        metadata
                    )  # Same metadata for all augmentations

            # Stack all images (clean + augmented for all samples in batch)
            all_images = torch.stack(batch_images).to(device)

            # Process metadata
            all_metadata = {}
            for key in batch_metadata_list[0].keys():
                all_metadata[key] = torch.stack(
                    [m[key] for m in batch_metadata_list]
                ).to(device)

            # Batch inference for all images at once
            all_outputs = model(all_images, all_metadata)

            # Reshape outputs: [batch_size * (1 + tta_rounds), num_classes]
            # -> [batch_size, (1 + tta_rounds), num_classes]
            outputs_per_sample = 1 + tta_rounds
            reshaped_outputs = all_outputs.view(
                current_batch_size, outputs_per_sample, -1
            )

            # Average predictions across all augmentations for each sample
            averaged_outputs = torch.mean(
                reshaped_outputs, dim=1
            )  # [batch_size, num_classes]
            averaged_probabilities = (
                torch.softmax(averaged_outputs, dim=1).cpu().numpy()
            )

            # Get final predictions
            predictions = averaged_outputs.argmax(1).cpu().numpy()

            # Store results for this batch
            for i, (filename, prediction, prob_row) in enumerate(
                zip(batch_filenames, predictions, averaged_probabilities)
            ):
                results.append((filename, prediction))
                prob_results.append([filename] + prob_row.tolist())

    # Save Predictions to CSV
    with open(output_csv_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([session_name])  # Write session name as the first line
        writer.writerows(results)  # Write filenames and predictions
    print(f"TTA Results saved to {output_csv_path}")

    # Save Probabilities to CSV
    with open(output_probs_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        # Write header with session name and class indices
        header = [session_name] + [f"class_{i}" for i in range(183)]
        writer.writerow(header)
        writer.writerows(prob_results)  # Write filenames and probabilities
    print(f"TTA Probabilities saved to {output_probs_path}")


if __name__ == "__main__":
    # Path to fungi images
    image_path = "/work3/monka/SummerSchool2025/FungiImages/"
    # Path to metadata file
    data_file = str("/work3/monka/SummerSchool2025/metadata.csv")

    # Session name: Change session name for every experiment!
    session = "EfficientNet_V2L_Multimodal_LateFusion"

    # Folder for results of this experiment based on session name:
    checkpoint_dir = os.path.join(f"/work3/monka/SummerSchool2025/results/{session}/")

    # Path to pre-trained image-only model
    pretrained_model_path = "/work3/monka/SummerSchool2025/results/EfficientNet_V2L_CrossEntropy_New/best_accuracy.pth"

    print(f"🍄 Starting Multimodal Training for {session}")
    print(f"📁 Results will be saved to: {checkpoint_dir}")
    print(f"🖼️  Using pre-trained model: {pretrained_model_path}")
    print("=" * 80)

    train_fungi_network(data_file, image_path, checkpoint_dir, pretrained_model_path)
    evaluate_network_on_test_set(data_file, image_path, checkpoint_dir, session)
    evaluate_network_on_test_set_with_tta(
        data_file, image_path, checkpoint_dir, session, tta_rounds=10
    )
    evaluate_network_on_test_set_with_tta(
        data_file, image_path, checkpoint_dir, session, tta_rounds=32
    )
