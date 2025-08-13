import os
import pandas as pd
import numpy as np
import json
from xgboost import XGBClassifier
import pickle
import warnings
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


class ConfigurableXGBoostFusion:
    """
    Configurable XGBoost fusion with toggleable feature types.
    Allows experimentation with different feature combinations.
    """

    def __init__(self, output_dir, feature_config=None):
        self.output_dir = output_dir
        self.model = None
        self.label_encoder = LabelEncoder()
        self.categorical_mappings = {}

        # Default feature configuration
        self.feature_config = {
            "use_image_features": True,
            "use_class_probabilities": True,
            "use_prediction_confidence": True,
            "use_prediction_entropy": True,
            "use_metadata_features": True,
            "metadata_multiply_image_fusion": False,
            "metadata_fusion_temperature": 2.0,
        }

        # Update with user config
        if feature_config:
            self.feature_config.update(feature_config)

        self.print_config()

    def print_config(self):
        """Print current feature configuration."""
        print("\n🔧 Feature Configuration:")
        print(f"  ✅ Image features: {self.feature_config['use_image_features']}")
        print(
            f"  ✅ Class probabilities: {self.feature_config['use_class_probabilities']}"
        )
        print(
            f"  ✅ Prediction confidence: {self.feature_config['use_prediction_confidence']}"
        )
        print(
            f"  ✅ Prediction entropy: {self.feature_config['use_prediction_entropy']}"
        )
        print(f"  ✅ Metadata features: {self.feature_config['use_metadata_features']}")
        print(
            f"  🔄 Metadata multiply image fusion: {self.feature_config['metadata_multiply_image_fusion']}"
        )
        if self.feature_config.get("metadata_multiply_image_fusion", False):
            print(
                f"  🌡️  Fusion temperature: {self.feature_config['metadata_fusion_temperature']}"
            )

    def load_data(self, features_dir):
        """Load the unified dataset with vector-based features."""
        dataset_file = os.path.join(features_dir, "unified_xgboost_dataset.pkl")
        df = pd.read_pickle(dataset_file)

        with open(os.path.join(features_dir, "feature_info.json"), "r") as f:
            feature_info = json.load(f)

        return df, feature_info

    def prepare_features(self, df, feature_info, is_train=True):
        """
        Configurable feature preparation based on feature_config flags.
        """
        feature_cols = []
        all_dataframes = []

        # Start with base metadata (always include core columns)
        base_df = df.drop(
            ["image_features_vector", "class_probabilities_vector"], axis=1
        ).copy()

        # 1. Image features (toggleable)
        if self.feature_config["use_image_features"]:
            print("  📸 Adding image features...")
            image_features = np.vstack(df["image_features_vector"].values)
            image_feature_cols = [
                f"image_feat_{i}" for i in range(image_features.shape[1])
            ]

            image_df = pd.DataFrame(
                image_features, columns=image_feature_cols, index=df.index
            )
            all_dataframes.append(image_df)
            feature_cols.extend(image_feature_cols)

        # 2. Class probabilities (toggleable)
        if self.feature_config["use_class_probabilities"]:
            print("  🎯 Adding class probabilities...")
            prob_features = np.vstack(df["class_probabilities_vector"].values)
            prob_feature_cols = [
                f"class_prob_{i}" for i in range(prob_features.shape[1])
            ]

            prob_df = pd.DataFrame(
                prob_features, columns=prob_feature_cols, index=df.index
            )
            all_dataframes.append(prob_df)
            feature_cols.extend(prob_feature_cols)

        # 3. Prediction confidence (toggleable)
        if self.feature_config["use_prediction_confidence"]:
            if "prediction_confidence" in base_df.columns:
                print("  📊 Adding prediction confidence...")
                feature_cols.append("prediction_confidence")

        # 4. Prediction entropy (toggleable)
        if self.feature_config["use_prediction_entropy"]:
            if "prediction_entropy" in base_df.columns:
                print("  🌀 Adding prediction entropy...")
                feature_cols.append("prediction_entropy")

        # Combine all feature dataframes
        if all_dataframes:
            features_combined = pd.concat([base_df] + all_dataframes, axis=1)
        else:
            features_combined = base_df

        # 5. Metadata features (toggleable)
        metadata_features = []
        if self.feature_config["use_metadata_features"]:
            print("  🌍 Adding metadata features...")

            # Categorical text features
            for col in ["Habitat", "Substrate"]:
                if col in features_combined.columns:
                    if is_train:
                        # Fit categories on training data and store them
                        features_combined[col] = features_combined[col].astype(
                            "category"
                        )
                        if not hasattr(self, "categorical_mappings"):
                            self.categorical_mappings = {}
                        self.categorical_mappings[col] = features_combined[
                            col
                        ].cat.categories
                    else:
                        # Use the same categories from training
                        if (
                            hasattr(self, "categorical_mappings")
                            and col in self.categorical_mappings
                        ):
                            features_combined[col] = pd.Categorical(
                                features_combined[col],
                                categories=self.categorical_mappings[col],
                            )
                        else:
                            # Fallback if mapping not found
                            raise ValueError(
                                f"Categorical mapping for {col} not found. Ensure model was trained with this column."
                            )
                    metadata_features.append(col)

            # Numerical features
            for col in ["Latitude", "Longitude"]:
                if col in features_combined.columns:
                    metadata_features.append(col)

            # Temporal features
            if "eventDate" in features_combined.columns:
                features_combined["eventDate_parsed"] = pd.to_datetime(
                    features_combined["eventDate"], errors="coerce"
                )
                features_combined["month"] = features_combined[
                    "eventDate_parsed"
                ].dt.month
                features_combined["year"] = features_combined[
                    "eventDate_parsed"
                ].dt.year
                features_combined["season"] = features_combined["month"].map(
                    {
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
                    }
                )
                metadata_features.extend(["month", "season", "year"])

        # Combine all selected features
        all_features = feature_cols + metadata_features

        if not all_features:
            raise ValueError(
                "No features selected! Please enable at least one feature type."
            )

        print(f"  🔢 Total features selected: {len(all_features)}")
        final_X = features_combined[all_features].copy()

        # Debug categorical encoding
        print(
            f"\n=== CATEGORICAL ENCODING DEBUG ({'TRAIN' if is_train else 'TEST'}) ==="
        )
        for col in ["Habitat", "Substrate"]:
            if col in final_X.columns:
                print(f"\n{col}:")
                print(f"  Raw unique values: {sorted(df[col].dropna().unique())}")
                print(f"  Number of raw unique: {df[col].nunique()}")

                if hasattr(final_X[col], "cat"):
                    print(
                        f"  Categories in final_X: {list(final_X[col].cat.categories)}"
                    )
                    print(
                        f"  Codes in final_X: {sorted(final_X[col].cat.codes.dropna().unique())}"
                    )
                    print(
                        f"  Missing values (coded as -1): {(final_X[col].cat.codes == -1).sum()}"
                    )

                    if (
                        hasattr(self, "categorical_mappings")
                        and col in self.categorical_mappings
                    ):
                        print(
                            f"  Stored training categories: {list(self.categorical_mappings[col])}"
                        )

                        # Check for categories in test that weren't in training
                        test_categories = set(df[col].dropna().unique())
                        train_categories = set(self.categorical_mappings[col])
                        new_categories = test_categories - train_categories
                        if new_categories:
                            print(
                                f"  ⚠️  NEW CATEGORIES IN TEST: {sorted(new_categories)}"
                            )
                        missing_categories = train_categories - test_categories
                        if missing_categories:
                            print(
                                f"  📝 MISSING CATEGORIES IN TEST: {sorted(missing_categories)}"
                            )
                else:
                    print("  Not categorical in final_X")
        print("=" * 60)

        return final_X

    def train(self, features_dir, debug_subset=None):
        """
        Train XGBoost model with configurable features.
        """
        print("Loading data...")
        df, feature_info = self.load_data(features_dir)
        train_df = df[df["dataset"] == "train"].copy()

        # Debug subset option
        if debug_subset is not None:
            print(f"🐛 DEBUG MODE: Using only first {debug_subset} samples")
            train_df = train_df.head(debug_subset)

        print(f"Training samples: {len(train_df)}")
        print(f"Classes: {train_df['true_label'].nunique()}")

        # Handle special metadata fusion strategy
        original_fusion_flag = self.feature_config.get(
            "metadata_multiply_image_fusion", False
        )

        if original_fusion_flag:
            print("🔄 Special metadata fusion strategy detected!")
            print(
                "   Training metadata-only model, fusion will happen during validation..."
            )
            # Force metadata-only configuration for training
            original_config = self.feature_config.copy()
            self.feature_config.update(
                {
                    "use_image_features": False,
                    "use_class_probabilities": False,
                    "use_prediction_confidence": False,
                    "use_prediction_entropy": False,
                    "use_metadata_features": True,
                }
            )

        # Split into train and validation sets
        print("Splitting data into train/validation...")
        train_split, val_split = train_test_split(
            train_df, test_size=0.2, random_state=42, stratify=train_df["true_label"]
        )

        print(f"Train split: {len(train_split)} samples")
        print(f"Validation split: {len(val_split)} samples")

        # Prepare features for training
        print("Preparing training features...")
        X_train = self.prepare_features(train_split, feature_info, is_train=True)
        y_train = train_split["true_label"].values

        # Prepare features for validation
        print("Preparing validation features...")
        X_val = self.prepare_features(val_split, feature_info, is_train=False)
        y_val = val_split["true_label"].values

        # Encode labels to consecutive integers
        print("Encoding labels...")
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_val_encoded = self.label_encoder.transform(y_val)

        print(f"Original train labels range: {y_train.min():.0f} - {y_train.max():.0f}")
        print(
            f"Encoded train labels range: {y_train_encoded.min()} - {y_train_encoded.max()}"
        )
        print(f"Train feature matrix: {X_train.shape}")
        print(f"Validation feature matrix: {X_val.shape}")
        print(f"Missing values in train: {X_train.isnull().sum().sum()}")
        print(f"Missing values in validation: {X_val.isnull().sum().sum()}")

        # XGBoost parameters
        params = {
            "objective": "multi:softprob",
            "num_class": len(np.unique(y_train_encoded)),
            "random_state": 42,
            "tree_method": "gpu_hist",
            "enable_categorical": True,
            "device": "cuda",
            "verbosity": 1,
        }

        # Train model
        print("Training XGBoost model...")
        self.model = XGBClassifier(**params)
        self.model.fit(X_train, y_train_encoded, verbose=True)

        # Evaluate on training set
        print("\nEvaluating model performance...")
        train_predictions_encoded = self.model.predict(X_train)
        train_predictions = self.label_encoder.inverse_transform(
            train_predictions_encoded
        )
        train_f1 = f1_score(y_train, train_predictions, average="weighted")
        train_accuracy = accuracy_score(y_train, train_predictions)

        print(f"Training F1 Score: {train_f1:.4f}")
        print(f"Training Accuracy: {train_accuracy:.4f}")

        # Evaluate on validation set
        print("\nValidation performance...")

        # Check if we need special metadata fusion
        if original_fusion_flag:
            print("🔄 Applying metadata multiply image fusion for validation...")

            # Get metadata-only predictions (smoothed)
            metadata_probabilities = self.model.predict_proba(X_val)
            # Apply smoothing to metadata probabilities (temperature scaling)
            temperature = self.feature_config.get("metadata_fusion_temperature", 2.0)
            metadata_probabilities_smoothed = np.exp(
                np.log(metadata_probabilities + 1e-8) / temperature
            )
            metadata_probabilities_smoothed = (
                metadata_probabilities_smoothed
                / metadata_probabilities_smoothed.sum(axis=1, keepdims=True)
            )

            # Get image-only predictions from class probabilities in the data
            # Restore original config temporarily to get image features
            temp_config = {
                "use_image_features": False,
                "use_class_probabilities": True,  # This contains image model predictions
                "use_prediction_confidence": False,
                "use_prediction_entropy": False,
                "use_metadata_features": False,
            }
            self.feature_config.update(temp_config)
            self.feature_config["metadata_multiply_image_fusion"] = False

            # Get image-only features (class probabilities)
            X_val_image = self.prepare_features(val_split, feature_info, is_train=False)

            # Extract image probabilities (they are the class_prob columns)
            image_prob_cols = [
                col for col in X_val_image.columns if col.startswith("class_prob_")
            ]
            if image_prob_cols:
                image_probabilities = X_val_image[image_prob_cols].values
                # Normalize to ensure they sum to 1
                image_probabilities = image_probabilities / image_probabilities.sum(
                    axis=1, keepdims=True
                )

                # Multiply image probabilities with smoothed metadata probabilities
                fused_probabilities = (
                    image_probabilities * metadata_probabilities_smoothed
                )
                fused_probabilities = fused_probabilities / fused_probabilities.sum(
                    axis=1, keepdims=True
                )

                # Get predictions from fused probabilities
                val_predictions_encoded = np.argmax(fused_probabilities, axis=1)
                val_predictions = self.label_encoder.inverse_transform(
                    val_predictions_encoded
                )

                print(f"  📊 Image probs shape: {image_probabilities.shape}")
                print(
                    f"  🌍 Metadata probs shape: {metadata_probabilities_smoothed.shape}"
                )
                print(f"  🔄 Fused probs shape: {fused_probabilities.shape}")
            else:
                print(
                    "  ⚠️  No image probabilities found, using metadata-only predictions"
                )
                val_predictions_encoded = self.model.predict(X_val)
                val_predictions = self.label_encoder.inverse_transform(
                    val_predictions_encoded
                )

            # Restore original config
            self.feature_config = original_config.copy()
            self.feature_config["metadata_multiply_image_fusion"] = original_fusion_flag

        else:
            # Standard validation evaluation
            val_predictions_encoded = self.model.predict(X_val)
            val_predictions = self.label_encoder.inverse_transform(
                val_predictions_encoded
            )

        val_f1 = f1_score(y_val, val_predictions, average="weighted")
        val_accuracy = accuracy_score(y_val, val_predictions)

        print(f"Validation F1 Score: {val_f1:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.4f}")

        # Feature importance
        importance_df = pd.DataFrame(
            {"feature": X_train.columns, "importance": self.model.feature_importances_}
        ).sort_values("importance", ascending=False)

        print("\nTop 10 most important features:")
        for i, row in importance_df.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")

        # Save everything
        self.save_model(importance_df, train_f1, train_accuracy, val_f1, val_accuracy)

        return {
            "message": "Training completed successfully",
            "train_f1_score": train_f1,
            "train_accuracy": train_accuracy,
            "val_f1_score": val_f1,
            "val_accuracy": val_accuracy,
            "feature_config": self.feature_config,
        }

    def predict_test(self, features_dir):
        """Generate test predictions."""
        if self.model is None:
            raise ValueError("Model not trained yet!")

        print("Loading test data...")
        df, feature_info = self.load_data(features_dir)
        test_df = df[df["dataset"] == "test"].copy()

        print(f"Test samples: {len(test_df)}")

        # Prepare features
        print("Preparing test features...")
        X_test = self.prepare_features(test_df, feature_info, is_train=False)

        # Check if we need special metadata fusion for test predictions
        if self.feature_config.get("metadata_multiply_image_fusion", False):
            print("🔄 Applying metadata multiply image fusion for test predictions...")

            # Get metadata-only predictions (smoothed)
            metadata_probabilities = self.model.predict_proba(X_test)
            # Apply smoothing to metadata probabilities (temperature scaling)
            temperature = self.feature_config.get("metadata_fusion_temperature", 2.0)
            metadata_probabilities_smoothed = np.exp(
                np.log(metadata_probabilities + 1e-8) / temperature
            )
            metadata_probabilities_smoothed = (
                metadata_probabilities_smoothed
                / metadata_probabilities_smoothed.sum(axis=1, keepdims=True)
            )

            # Get image-only predictions from class probabilities in the data
            # Temporarily change config to get image features
            temp_config = {
                "use_image_features": False,
                "use_class_probabilities": True,  # This contains image model predictions
                "use_prediction_confidence": False,
                "use_prediction_entropy": False,
                "use_metadata_features": False,
            }
            original_config = self.feature_config.copy()
            self.feature_config.update(temp_config)
            self.feature_config["metadata_multiply_image_fusion"] = False

            # Get image-only features (class probabilities)
            X_test_image = self.prepare_features(test_df, feature_info, is_train=False)

            # Extract image probabilities (they are the class_prob columns)
            image_prob_cols = [
                col for col in X_test_image.columns if col.startswith("class_prob_")
            ]
            if image_prob_cols:
                image_probabilities = X_test_image[image_prob_cols].values
                # Normalize to ensure they sum to 1
                image_probabilities = image_probabilities / image_probabilities.sum(
                    axis=1, keepdims=True
                )

                # Multiply image probabilities with smoothed metadata probabilities
                fused_probabilities = (
                    image_probabilities * metadata_probabilities_smoothed
                )
                fused_probabilities = fused_probabilities / fused_probabilities.sum(
                    axis=1, keepdims=True
                )

                # Get predictions from fused probabilities
                predictions_encoded = np.argmax(fused_probabilities, axis=1)
                predictions = self.label_encoder.inverse_transform(predictions_encoded)
                probabilities = fused_probabilities

                print(f"  📊 Image probs shape: {image_probabilities.shape}")
                print(
                    f"  🌍 Metadata probs shape: {metadata_probabilities_smoothed.shape}"
                )
                print(f"  🔄 Fused probs shape: {fused_probabilities.shape}")
            else:
                print(
                    "  ⚠️  No image probabilities found, using metadata-only predictions"
                )
                predictions_encoded = self.model.predict(X_test)
                predictions = self.label_encoder.inverse_transform(predictions_encoded)
                probabilities = self.model.predict_proba(X_test)

            # Restore original config
            self.feature_config = original_config.copy()

        else:
            # Standard test prediction
            print("Generating predictions...")
            predictions_encoded = self.model.predict(X_test)
            predictions = self.label_encoder.inverse_transform(predictions_encoded)
            probabilities = self.model.predict_proba(X_test)

        # Create submission
        results_df = pd.DataFrame(
            {
                "filename": test_df["filename"].values,
                "predicted_label": predictions.astype(int),
                "confidence": np.max(probabilities, axis=1),
            }
        )

        config_suffix = self.get_config_suffix()

        # Save detailed test results CSV (like the original script)
        detailed_results_file = os.path.join(
            self.output_dir, f"test_results{config_suffix}.csv"
        )
        results_df.to_csv(detailed_results_file, index=False)
        print(f"Detailed test results saved to: {detailed_results_file}")

        # Save in competition format (submission file)
        submission_file = os.path.join(
            self.output_dir, f"test_predictions{config_suffix}.csv"
        )
        with open(submission_file, "w") as f:
            f.write(f"XGBoost_Configurable_Fusion{config_suffix}\n")
            for _, row in results_df.iterrows():
                f.write(f"{row['filename']},{row['predicted_label']}\n")

        print(f"Competition submission saved to: {submission_file}")
        print(f"Average confidence: {results_df['confidence'].mean():.4f}")

        return results_df

    def get_config_suffix(self):
        """Generate suffix based on enabled features."""
        config = self.feature_config
        suffix_parts = []

        if config.get("metadata_multiply_image_fusion", False):
            suffix_parts.append(
                f"META_IMG_FUSION_{config['metadata_fusion_temperature']:.2f}"
            )
        else:
            if config["use_image_features"]:
                suffix_parts.append("IMG")
            if config["use_class_probabilities"]:
                suffix_parts.append("PROB")
            if config["use_prediction_confidence"]:
                suffix_parts.append("CONF")
            if config["use_prediction_entropy"]:
                suffix_parts.append("ENT")
            if config["use_metadata_features"]:
                suffix_parts.append("META")

        return "_" + "+".join(suffix_parts) if suffix_parts else "_NONE"

    def save_model(
        self, importance_df, train_f1, train_accuracy, val_f1=None, val_accuracy=None
    ):
        """Save model and results with feature configuration."""
        os.makedirs(self.output_dir, exist_ok=True)

        config_suffix = self.get_config_suffix()

        # Save model and label encoder
        model_file = os.path.join(
            self.output_dir, f"xgboost_fusion_model{config_suffix}.pkl"
        )
        with open(model_file, "wb") as f:
            pickle.dump(
                {
                    "model": self.model,
                    "label_encoder": self.label_encoder,
                    "feature_config": self.feature_config,
                    "categorical_mappings": self.categorical_mappings,
                },
                f,
            )

        # Save feature importance
        importance_file = os.path.join(
            self.output_dir, f"feature_importance{config_suffix}.csv"
        )
        importance_df.to_csv(importance_file, index=False)

        # Save metrics with configuration
        metrics_file = os.path.join(
            self.output_dir, f"training_metrics{config_suffix}.json"
        )
        metrics_data = {
            "feature_config": self.feature_config,
            "train_f1_score": float(train_f1),
            "train_accuracy": float(train_accuracy),
        }

        if val_f1 is not None and val_accuracy is not None:
            metrics_data.update(
                {
                    "val_f1_score": float(val_f1),
                    "val_accuracy": float(val_accuracy),
                }
            )

        with open(metrics_file, "w") as f:
            json.dump(metrics_data, f, indent=2)

        print(f"Model and results saved to: {self.output_dir}")


def run_experiment(
    features_dir, output_dir, feature_config, experiment_name, debug_subset=None
):
    """Run a single experiment with given feature configuration."""
    print(f"\n🧪 Experiment: {experiment_name}")
    print("=" * 60)

    # Initialize with specific config
    trainer = ConfigurableXGBoostFusion(output_dir, feature_config)

    # Train
    results = trainer.train(features_dir, debug_subset=debug_subset)

    # Predict test set
    test_results = trainer.predict_test(features_dir)

    print(f"\n📊 {experiment_name} Results:")
    print(f"  Train F1: {results['train_f1_score']:.4f}")
    print(f"  Train Acc: {results['train_accuracy']:.4f}")
    print(f"  Val F1: {results['val_f1_score']:.4f}")
    print(f"  Val Acc: {results['val_accuracy']:.4f}")

    return results


def main():
    """Run experiments with different feature combinations."""
    features_dir = "/work3/monka/SummerSchool2025/results/EfficientNet_V2L_CrossEntropy/extracted_features/"
    output_dir = "/work3/monka/SummerSchool2025/results/XGBoost_Configurable_Experiments_WithUpdatedMetadata_0/"

    print("🍄 Configurable XGBoost Fusion - Feature Ablation Study")
    print("=" * 70)

    # Define different configurations to test
    experiments = [
        # {
        #     "name": "All Features",
        #     "config": {
        #         "use_image_features": True,
        #         "use_class_probabilities": True,
        #         "use_prediction_confidence": True,
        #         "use_prediction_entropy": True,
        #         "use_metadata_features": True,
        #     },
        # },
        # {
        #     "name": "Only Image Features",
        #     "config": {
        #         "use_image_features": True,
        #         "use_class_probabilities": False,
        #         "use_prediction_confidence": False,
        #         "use_prediction_entropy": False,
        #         "use_metadata_features": False,
        #     },
        # },
        # {
        #     "name": "Image + Metadata",
        #     "config": {
        #         "use_image_features": True,
        #         "use_class_probabilities": False,
        #         "use_prediction_confidence": False,
        #         "use_prediction_entropy": False,
        #         "use_metadata_features": True,
        #     },
        # },
        {
            "name": "Only Metadata",
            "config": {
                "use_image_features": False,
                "use_class_probabilities": False,
                "use_prediction_confidence": False,
                "use_prediction_entropy": False,
                "use_metadata_features": True,
            },
        },
        {
            "name": "Metadata Multiply Image Fusion 0.5",
            "config": {
                "use_image_features": False,  # Will be forced in the fusion strategy
                "use_class_probabilities": False,
                "use_prediction_confidence": False,
                "use_prediction_entropy": False,
                "use_metadata_features": True,
                "metadata_multiply_image_fusion": True,
                "metadata_fusion_temperature": 0.5,  # Configurable temperature for smoothing
            },
        },
        {
            "name": "Metadata Multiply Image Fusion 1.0",
            "config": {
                "use_image_features": False,  # Will be forced in the fusion strategy
                "use_class_probabilities": False,
                "use_prediction_confidence": False,
                "use_prediction_entropy": False,
                "use_metadata_features": True,
                "metadata_multiply_image_fusion": True,
                "metadata_fusion_temperature": 1.0,  # Configurable temperature for smoothing
            },
        },
        {
            "name": "Metadata Multiply Image Fusion 2.0",
            "config": {
                "use_image_features": False,  # Will be forced in the fusion strategy
                "use_class_probabilities": False,
                "use_prediction_confidence": False,
                "use_prediction_entropy": False,
                "use_metadata_features": True,
                "metadata_multiply_image_fusion": True,
                "metadata_fusion_temperature": 2.0,  # Configurable temperature for smoothing
            },
        },
        {
            "name": "Metadata Multiply Image Fusion 4.0",
            "config": {
                "use_image_features": False,  # Will be forced in the fusion strategy
                "use_class_probabilities": False,
                "use_prediction_confidence": False,
                "use_prediction_entropy": False,
                "use_metadata_features": True,
                "metadata_multiply_image_fusion": True,
                "metadata_fusion_temperature": 4.0,  # Configurable temperature for smoothing
            },
        },
    ]

    # Run experiments
    all_results = {}
    debug_subset = None  # Use small subset for quick testing

    for experiment in experiments:
        try:
            results = run_experiment(
                features_dir=features_dir,
                output_dir=output_dir,
                feature_config=experiment["config"],
                experiment_name=experiment["name"],
                debug_subset=debug_subset,
            )
            all_results[experiment["name"]] = results
        except Exception as e:
            print(f"❌ Experiment '{experiment['name']}' failed: {e}")

    # Summary
    print("\n🏆 EXPERIMENT SUMMARY")
    print("=" * 70)
    print(
        f"{'Experiment':<25} {'Train F1':<10} {'Val F1':<10} {'Train Acc':<10} {'Val Acc':<10}"
    )
    print("-" * 70)

    for name, results in all_results.items():
        print(
            f"{name:<25} {results['train_f1_score']:<10.4f} {results['val_f1_score']:<10.4f} "
            f"{results['train_accuracy']:<10.4f} {results['val_accuracy']:<10.4f}"
        )


if __name__ == "__main__":
    main()
