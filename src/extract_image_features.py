import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from albumentations import Compose, Normalize, Resize
from albumentations.pytorch import ToTensorV2
from torchvision import models
import tqdm
import numpy as np
from PIL import Image
import pickle


def get_transforms_for_inference():
    """
    Return transforms for inference (same as validation transforms).
    """
    # width, height = 224, 224
    width, height = 224 * 2, 224 * 2
    return Compose(
        [
            Resize(height=height, width=width),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
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


class FeatureExtractor(nn.Module):
    """
    Wrapper around EfficientNet to extract features from penultimate layer.
    """

    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        self.features = model.features
        self.avgpool = model.avgpool
        self.dropout = model.classifier[0]  # Dropout layer
        self.classifier = model.classifier[1]  # Linear layer

    def forward(self, x):
        # Extract features from the penultimate layer (before final classification)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        penultimate_features = self.dropout(
            x
        )  # Features after dropout, before classification

        # Also get final predictions
        predictions = self.classifier(penultimate_features)

        return penultimate_features, predictions


def extract_features_from_model(
    data_file,
    image_path,
    checkpoint_dir,
    output_dir,
    dataset_type="all",
):
    """
    Extract features from trained model for all datasets (train, test, final).
    Creates a unified dataset ready for XGBoost training.

    Args:
        data_file: Path to metadata CSV file
        image_path: Path to images directory
        checkpoint_dir: Directory containing trained model checkpoints
        output_dir: Directory to save extracted features
        dataset_type: Which dataset to process ("train", "test", "final", or "all")
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # model = models.efficientnet_b2(pretrained=True)
    model = models.efficientnet_v2_l(pretrained=True)
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.classifier[1].in_features, 183),  # Number of classes
    )

    # Load trained weights
    best_trained_model = os.path.join(checkpoint_dir, "best_accuracy.pth")
    if not os.path.exists(best_trained_model):
        raise FileNotFoundError(f"Trained model not found at {best_trained_model}")

    model.load_state_dict(torch.load(best_trained_model, map_location=device))
    model.to(device)

    # Create feature extractor
    feature_extractor = FeatureExtractor(model)
    feature_extractor.eval()

    # Load metadata once
    df = pd.read_csv(data_file)

    # Determine which datasets to process
    datasets_to_process = []
    if dataset_type == "all":
        datasets_to_process = ["train", "test", "final"]
    else:
        datasets_to_process = [dataset_type]

    # Store all data for unified dataset
    unified_data = []

    for dataset in datasets_to_process:
        print(f"\nProcessing {dataset} dataset...")

        # Filter data for current dataset
        dataset_df = df[df["filename_index"].str.startswith(f"fungi_{dataset}")].copy()

        if len(dataset_df) == 0:
            print(f"No data found for {dataset} dataset. Skipping...")
            continue

        print(f"Found {len(dataset_df)} samples in {dataset} dataset")

        # Create dataset and dataloader
        dataset_obj = FungiDataset(
            dataset_df, image_path, transform=get_transforms_for_inference()
        )
        dataloader = DataLoader(
            dataset_obj, batch_size=64, shuffle=False, num_workers=4
        )

        # Extract features
        batch_results = []

        with torch.no_grad():
            for images, labels, filenames in tqdm.tqdm(
                dataloader, desc=f"Extracting {dataset} features"
            ):
                images = images.to(device)

                # Extract features and predictions
                features, predictions = feature_extractor(images)
                probabilities = torch.softmax(predictions, dim=1)

                # Convert to numpy
                features_np = features.cpu().numpy()
                predictions_np = predictions.argmax(1).cpu().numpy()
                probabilities_np = probabilities.cpu().numpy()

                # Store batch results
                for i, filename in enumerate(filenames):
                    batch_results.append(
                        {
                            "filename": filename,
                            "dataset": dataset,
                            "image_features": features_np[i],
                            "predicted_class": predictions_np[i],
                            "class_probabilities": probabilities_np[i],
                            "true_label": labels[i].item() if labels[i] != -1 else None,
                        }
                    )

        print(f"Extracted features for {len(batch_results)} samples")
        unified_data.extend(batch_results)

    # Create unified dataset
    if unified_data:
        create_unified_xgboost_dataset(unified_data, df, output_dir)


def create_unified_xgboost_dataset(unified_data, original_df, output_dir):
    """
    Create a unified dataset perfect for XGBoost training with features stored as vector columns.
    This approach stores features as numpy arrays in DataFrame columns rather than expanding
    each feature dimension as a separate column, making it more efficient and cleaner.
    """
    print(f"\nCreating unified XGBoost dataset with {len(unified_data)} samples...")

    # Convert to DataFrame
    rows = []

    for item in unified_data:
        filename = item["filename"]

        # Get metadata for this sample
        metadata_row = original_df[original_df["filename_index"] == filename].iloc[0]

        # Create row with all data - store features as vector columns
        row = {
            "filename": filename,
            "dataset": item["dataset"],
            "true_label": item["true_label"],
            "predicted_class": item["predicted_class"],
            # Metadata features
            "Habitat": metadata_row.get("Habitat", ""),
            "Latitude": metadata_row.get("Latitude", np.nan),
            "Longitude": metadata_row.get("Longitude", np.nan),
            "Substrate": metadata_row.get("Substrate", ""),
            "eventDate": metadata_row.get("eventDate", ""),
            # Store feature vectors as single columns (not expanded)
            "image_features_vector": item["image_features"],  # Store as numpy array
            "class_probabilities_vector": item[
                "class_probabilities"
            ],  # Store as numpy array
            # Derived features
            "prediction_confidence": np.max(item["class_probabilities"]),
            "prediction_entropy": -np.sum(
                (item["class_probabilities"] + 1e-15)
                * np.log(item["class_probabilities"] + 1e-15)
            ),
            # Feature dimensions for reference
            "image_features_dim": len(item["image_features"]),
            "num_classes": len(item["class_probabilities"]),
        }

        rows.append(row)

    # Create final DataFrame
    unified_df = pd.DataFrame(rows)

    print(f"Unified dataset shape: {unified_df.shape}")
    print(
        f"Image features dimension: {unified_df['image_features_dim'].iloc[0] if len(unified_df) > 0 else 'N/A'}"
    )
    print(
        f"Number of classes: {unified_df['num_classes'].iloc[0] if len(unified_df) > 0 else 'N/A'}"
    )

    # Save as multiple formats for convenience

    # 1. Pickle (Python-specific but handles numpy arrays in columns well)
    pickle_file = os.path.join(output_dir, "unified_xgboost_dataset.pkl")
    unified_df.to_pickle(pickle_file)
    print(f"Saved unified dataset as Pickle: {pickle_file}")

    # 2. Parquet (efficient, but need to handle numpy arrays)
    # Convert numpy arrays to lists for parquet compatibility
    df_for_parquet = unified_df.copy()
    df_for_parquet["image_features_vector"] = df_for_parquet[
        "image_features_vector"
    ].apply(lambda x: x.tolist())
    df_for_parquet["class_probabilities_vector"] = df_for_parquet[
        "class_probabilities_vector"
    ].apply(lambda x: x.tolist())

    parquet_file = os.path.join(output_dir, "unified_xgboost_dataset.parquet")
    df_for_parquet.to_parquet(parquet_file, index=False)
    print(f"Saved unified dataset as Parquet: {parquet_file}")

    # 3. Also save in a format that's easy to use with XGBoost
    # Create a version where features are properly formatted for ML
    xgboost_ready_file = os.path.join(output_dir, "xgboost_ready_dataset.pkl")

    # Prepare feature matrices
    xgboost_data = {}
    for dataset_split in unified_df["dataset"].unique():
        split_data = unified_df[unified_df["dataset"] == dataset_split].copy()

        if len(split_data) > 0:
            # Stack image features into a 2D array
            image_features_matrix = np.vstack(
                split_data["image_features_vector"].values
            )
            prob_features_matrix = np.vstack(
                split_data["class_probabilities_vector"].values
            )

            # Combine all features
            numerical_features = (
                split_data[
                    [
                        "Latitude",
                        "Longitude",
                        "prediction_confidence",
                        "prediction_entropy",
                    ]
                ]
                .fillna(0)
                .values
            )

            # Full feature matrix combining image features, probabilities, and numerical features
            full_features = np.hstack(
                [image_features_matrix, prob_features_matrix, numerical_features]
            )

            xgboost_data[dataset_split] = {
                "features": full_features,
                "labels": split_data["true_label"].values,
                "filenames": split_data["filename"].values,
                "predicted_classes": split_data["predicted_class"].values,
                "image_features": image_features_matrix,
                "class_probabilities": prob_features_matrix,
                "metadata": split_data[
                    ["Habitat", "Substrate", "eventDate", "Latitude", "Longitude"]
                ].values,
                "feature_names": {
                    "image_features": [
                        f"image_feat_{i}" for i in range(image_features_matrix.shape[1])
                    ],
                    "class_probabilities": [
                        f"class_prob_{i}" for i in range(prob_features_matrix.shape[1])
                    ],
                    "numerical_features": [
                        "Latitude",
                        "Longitude",
                        "prediction_confidence",
                        "prediction_entropy",
                    ],
                },
            }

    # Save XGBoost-ready data
    with open(xgboost_ready_file, "wb") as f:
        pickle.dump(xgboost_data, f)
    print(f"Saved XGBoost-ready dataset: {xgboost_ready_file}")

    # 4. Save feature metadata for easy reference
    feature_info = {
        "dataset_format": "vector_columns",
        "vector_columns": {
            "image_features_vector": "numpy array with image features from EfficientNet-B2",
            "class_probabilities_vector": "numpy array with class probabilities",
        },
        "scalar_features": [
            "Habitat",
            "Latitude",
            "Longitude",
            "Substrate",
            "eventDate",
            "prediction_confidence",
            "prediction_entropy",
        ],
        "target": "true_label",
        "identifier": "filename",
        "dataset_split": "dataset",
        "dimensions": {
            "image_features": int(unified_df["image_features_dim"].iloc[0])
            if len(unified_df) > 0
            else 0,
            "num_classes": int(unified_df["num_classes"].iloc[0])
            if len(unified_df) > 0
            else 0,
        },
    }

    feature_info_file = os.path.join(output_dir, "feature_info.json")
    import json

    with open(feature_info_file, "w") as f:
        json.dump(feature_info, f, indent=2)
    print(f"Saved feature info: {feature_info_file}")

    # Print summary statistics
    print("\nDataset Summary:")
    print(f"  Total samples: {len(unified_df)}")
    print(f"  Train samples: {len(unified_df[unified_df['dataset'] == 'train'])}")
    print(f"  Test samples: {len(unified_df[unified_df['dataset'] == 'test'])}")
    print(f"  Final samples: {len(unified_df[unified_df['dataset'] == 'final'])}")
    print(f"  Image features dimension: {feature_info['dimensions']['image_features']}")
    print(f"  Number of classes: {feature_info['dimensions']['num_classes']}")
    print(f"  Scalar features: {len(feature_info['scalar_features'])}")

    return unified_df


def load_unified_dataset(output_dir, format="pickle"):
    """
    Load the unified XGBoost dataset with vector-type feature columns.

    Args:
        output_dir: Directory containing the dataset
        format: 'pickle' or 'parquet'

    Returns:
        DataFrame with vector-type feature columns
    """
    if format == "pickle":
        file_path = os.path.join(output_dir, "unified_xgboost_dataset.pkl")
        return pd.read_pickle(file_path)
    elif format == "parquet":
        file_path = os.path.join(output_dir, "unified_xgboost_dataset.parquet")
        df = pd.read_parquet(file_path)
        # Convert lists back to numpy arrays
        df["image_features_vector"] = df["image_features_vector"].apply(np.array)
        df["class_probabilities_vector"] = df["class_probabilities_vector"].apply(
            np.array
        )
        return df
    else:
        raise ValueError("Format must be 'pickle' or 'parquet'")


def load_xgboost_ready_dataset(output_dir):
    """
    Load the XGBoost-ready dataset with properly formatted feature matrices.

    Args:
        output_dir: Directory containing the dataset

    Returns:
        Dictionary with train/test/final splits, each containing feature matrices and labels
    """
    xgboost_file = os.path.join(output_dir, "xgboost_ready_dataset.pkl")
    with open(xgboost_file, "rb") as f:
        return pickle.load(f)


def load_feature_info(output_dir):
    """Load the feature information dictionary."""
    import json

    feature_info_file = os.path.join(output_dir, "feature_info.json")
    with open(feature_info_file, "r") as f:
        return json.load(f)


def prepare_features_for_xgboost(df, feature_types=None):
    """
    Extract and prepare features from the vector-based dataset for XGBoost training.

    Args:
        df: DataFrame with vector-type feature columns
        feature_types: List of feature types to include ('image', 'probabilities', 'numerical', 'all')
                      Default is 'all'

    Returns:
        Tuple of (feature_matrix, feature_names)
    """
    if feature_types is None:
        feature_types = ["all"]

    feature_matrices = []
    feature_names = []

    if "all" in feature_types or "image" in feature_types:
        # Extract image features
        image_features = np.vstack(df["image_features_vector"].values)
        feature_matrices.append(image_features)
        feature_names.extend(
            [f"image_feat_{i}" for i in range(image_features.shape[1])]
        )

    if "all" in feature_types or "probabilities" in feature_types:
        # Extract class probabilities
        prob_features = np.vstack(df["class_probabilities_vector"].values)
        feature_matrices.append(prob_features)
        feature_names.extend([f"class_prob_{i}" for i in range(prob_features.shape[1])])

    if "all" in feature_types or "numerical" in feature_types:
        # Extract numerical features
        numerical_cols = [
            "Latitude",
            "Longitude",
            "prediction_confidence",
            "prediction_entropy",
        ]
        numerical_features = df[numerical_cols].fillna(0).values
        feature_matrices.append(numerical_features)
        feature_names.extend(numerical_cols)

    # Combine all selected features
    if len(feature_matrices) == 1:
        final_features = feature_matrices[0]
    else:
        final_features = np.hstack(feature_matrices)

    return final_features, feature_names


if __name__ == "__main__":
    # Configuration
    image_path = "/work3/monka/SummerSchool2025/FungiImages/"
    data_file = "/work3/monka/SummerSchool2025/metadata.csv"
    checkpoint_dir = (
        "/work3/monka/SummerSchool2025/results/EfficientNet_V2L_CrossEntropy/"
    )
    output_dir = f"{checkpoint_dir}/extracted_features/"

    # Extract features and create unified dataset
    extract_features_from_model(
        data_file=data_file,
        image_path=image_path,
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
        dataset_type="all",  # Extract for train, test, and final datasets
    )

    # Example of how to load and use the unified dataset
    print("\n" + "=" * 60)
    print("Example: Loading and using the unified dataset")
    print("=" * 60)

    try:
        # Load the unified dataset
        df = load_unified_dataset(output_dir, format="pickle")
        feature_info = load_feature_info(output_dir)

        print(f"Loaded unified dataset with shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")

        # Example: Prepare data for XGBoost
        train_data = df[df["dataset"] == "train"].copy()
        test_data = df[df["dataset"] == "test"].copy()

        if len(train_data) > 0:
            print(f"\nTrain data: {len(train_data)} samples")
            print(f"Test data: {len(test_data)} samples")

            # Show feature availability
            print("\nFeature info:")
            print(
                f"  Image features dimension: {feature_info['dimensions']['image_features']}"
            )
            print(f"  Number of classes: {feature_info['dimensions']['num_classes']}")
            print(f"  Scalar features: {len(feature_info['scalar_features'])}")

            # Example: Prepare features for XGBoost using different feature combinations
            print("\nPreparing features for XGBoost...")

            # Using all features
            X_train_all, feature_names_all = prepare_features_for_xgboost(
                train_data, ["all"]
            )
            y_train = train_data["true_label"].values

            print(f"All features matrix shape: {X_train_all.shape}")
            print(f"Total feature names: {len(feature_names_all)}")

            # Using only image features
            X_train_img, feature_names_img = prepare_features_for_xgboost(
                train_data, ["image"]
            )
            print(f"Image features only matrix shape: {X_train_img.shape}")

            # Check for missing values
            print(f"Missing values in feature matrix: {np.isnan(X_train_all).sum()}")

            # Example: Load XGBoost-ready format
            print("\nLoading XGBoost-ready format...")
            xgb_data = load_xgboost_ready_dataset(output_dir)

            if "train" in xgb_data:
                print(
                    f"XGBoost train features shape: {xgb_data['train']['features'].shape}"
                )
                print(
                    f"XGBoost train labels shape: {xgb_data['train']['labels'].shape}"
                )

    except Exception as e:
        print(f"Could not load unified dataset: {e}")
        print("This is expected if the extraction hasn't been run yet.")
