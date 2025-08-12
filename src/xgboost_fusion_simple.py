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


class SimpleXGBoostFusion:
    """
    Simple but effective XGBoost fusion for multimodal fungi classification.
    Kaggle grandmaster style: minimal preprocessing, let XGBoost handle the rest.
    """

    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.model = None
        self.label_encoder = LabelEncoder()
        self.categorical_mappings = {}

    def load_data(self, features_dir):
        """Load the unified dataset with vector-based features."""
        dataset_file = os.path.join(features_dir, "unified_xgboost_dataset.pkl")
        df = pd.read_pickle(dataset_file)

        with open(os.path.join(features_dir, "feature_info.json"), "r") as f:
            feature_info = json.load(f)

        return df, feature_info

    def prepare_features(self, df, feature_info, is_train=True):
        """
        Minimal feature preparation - let XGBoost handle the complexity.
        Works with vector-based feature columns.
        """
        print(df)

        # Extract features from vector columns
        image_features = np.vstack(df["image_features_vector"].values)
        prob_features = np.vstack(df["class_probabilities_vector"].values)

        # Convert to DataFrame for easier handling
        image_feature_cols = [f"image_feat_{i}" for i in range(image_features.shape[1])]
        prob_feature_cols = [f"class_prob_{i}" for i in range(prob_features.shape[1])]

        # Create feature DataFrame
        features_df = pd.DataFrame(
            image_features, columns=image_feature_cols, index=df.index
        )
        prob_df = pd.DataFrame(prob_features, columns=prob_feature_cols, index=df.index)

        # Combine with original metadata
        X = pd.concat(
            [
                df.drop(
                    ["image_features_vector", "class_probabilities_vector"], axis=1
                ),
                features_df,
                prob_df,
            ],
            axis=1,
        )

        # Start with image features, probabilities, and derived features
        feature_cols = (
            image_feature_cols
            + prob_feature_cols
            + ["prediction_confidence", "prediction_entropy"]
        )

        # Add simple metadata features
        metadata_features = []

        # Add categorical text features with consistent encoding
        for col in ["Habitat", "Substrate"]:
            if col in X.columns:
                if is_train:
                    # Fit categories on training data and store them
                    X[col] = X[col].astype("category")
                    if not hasattr(self, "categorical_mappings"):
                        self.categorical_mappings = {}
                    self.categorical_mappings[col] = X[col].cat.categories
                else:
                    # Use the same categories from training
                    if (
                        hasattr(self, "categorical_mappings")
                        and col in self.categorical_mappings
                    ):
                        X[col] = pd.Categorical(
                            X[col], categories=self.categorical_mappings[col]
                        )
                    else:
                        # Fallback if mapping not found
                        raise ValueError(
                            f"Categorical mapping for {col} not found. Ensure model was trained with this column."
                        )

                metadata_features.append(col)

        # Add numerical features directly
        for col in ["Latitude", "Longitude"]:
            if col in X.columns:
                metadata_features.append(col)

        # Simple temporal features
        if "eventDate" in X.columns:
            X["eventDate_parsed"] = pd.to_datetime(X["eventDate"], errors="coerce")
            X["month"] = X["eventDate_parsed"].dt.month
            X["year"] = X["eventDate_parsed"].dt.year
            X["season"] = X["month"].map(
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

        # Combine all features
        all_features = feature_cols + metadata_features
        final_X = X[all_features].copy()

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
        Train XGBoost model - simple and fast.

        Args:
            features_dir: Directory containing the features
            debug_subset: If provided, train on only first N samples for debugging
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

        # Split into train and validation sets
        print("Splitting data into train/validation...")
        train_split, val_split = train_test_split(
            train_df, test_size=0.01, random_state=42, stratify=train_df["true_label"]
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

        # Encode labels to consecutive integers (required by XGBoost)
        print("Encoding labels...")
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        # Transform validation labels using the same encoder
        y_val_encoded = self.label_encoder.transform(y_val)

        print(f"Original train labels range: {y_train.min():.0f} - {y_train.max():.0f}")
        print(
            f"Encoded train labels range: {y_train_encoded.min()} - {y_train_encoded.max()}"
        )

        print(f"Train feature matrix: {X_train.shape}")
        print(f"Validation feature matrix: {X_val.shape}")
        print(f"Missing values in train: {X_train.isnull().sum().sum()}")
        print(f"Missing values in validation: {X_val.isnull().sum().sum()}")

        # XGBoost parameters (Kaggle competition style)
        params = {
            "objective": "multi:softprob",
            "num_class": len(
                np.unique(y_train_encoded)
            ),  # Use actual number of classes
            # "max_depth": 6,  # Conservative depth
            # "learning_rate": 0.15,  # Higher for faster convergence
            # "n_estimators": 500,  # Reduced for faster training
            # "subsample": 0.8,
            # "colsample_bytree": 0.8,
            # "reg_alpha": 0.1,
            # "reg_lambda": 1.0,
            "random_state": 42,
            "tree_method": "gpu_hist",  # Use GPU if available
            "enable_categorical": True,  # Enable native categorical support
            "device": "cuda",  # Use CUDA for GPU acceleration
            "verbosity": 2,  # Verbose output
        }

        # Train model
        print("Training XGBoost model...")
        self.model = XGBClassifier(**params)
        self.model.fit(X_train, y_train_encoded, verbose=True)  # Use encoded labels

        # Evaluate on training set
        print("\nEvaluating model performance...")
        train_predictions_encoded = self.model.predict(X_train)
        # Decode predictions back to original labels for evaluation
        train_predictions = self.label_encoder.inverse_transform(
            train_predictions_encoded
        )
        train_f1 = f1_score(y_train, train_predictions, average="weighted")
        train_accuracy = accuracy_score(y_train, train_predictions)

        print(f"Training F1 Score: {train_f1:.4f}")
        print(f"Training Accuracy: {train_accuracy:.4f}")

        # Evaluate on validation set
        print("\nValidation performance...")
        val_predictions_encoded = self.model.predict(X_val)
        val_predictions = self.label_encoder.inverse_transform(val_predictions_encoded)
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
        }

    def predict_test(self, features_dir):
        """Generate test predictions."""
        if self.model is None:
            raise ValueError("Model not trained yet!")

        print("Loading test data...")
        df, feature_info = self.load_data(features_dir)
        test_df = df[df["dataset"] == "test"].copy()

        # Prepare features (using fitted encoders)
        X_test = self.prepare_features(test_df, feature_info, is_train=False)

        # Predict
        print("Generating predictions...")
        predictions_encoded = self.model.predict(X_test)
        # Decode predictions back to original label space
        predictions = self.label_encoder.inverse_transform(predictions_encoded)
        probabilities = self.model.predict_proba(X_test)

        # Create submission
        results_df = pd.DataFrame(
            {
                "filename": test_df["filename"].values,
                "predicted_label": predictions.astype(int),  # Convert to int
                "confidence": np.max(probabilities, axis=1),
            }
        )

        # Save in competition format
        output_file = os.path.join(self.output_dir, "test_predictions.csv")
        with open(output_file, "w") as f:
            f.write("XGBoost_Full_Fusion\n")
            for _, row in results_df.iterrows():
                f.write(f"{row['filename']},{row['predicted_label']}\n")

        print(f"Predictions saved to: {output_file}")
        print(f"Average confidence: {results_df['confidence'].mean():.4f}")

        return results_df

    def save_model(
        self,
        importance_df,
        train_f1,
        train_accuracy,
        val_f1=None,
        val_accuracy=None,
    ):
        """Save model and results."""
        os.makedirs(self.output_dir, exist_ok=True)

        # Save model and label encoder
        model_file = os.path.join(self.output_dir, "xgboost_fusion_model.pkl")
        with open(model_file, "wb") as f:
            pickle.dump(
                {
                    "model": self.model,
                    "label_encoder": self.label_encoder,
                    "categorical_mappings": self.categorical_mappings,
                },
                f,
            )

        # Save feature importance
        importance_file = os.path.join(self.output_dir, "feature_importance.csv")
        importance_df.to_csv(importance_file, index=False)

        # Save metrics
        metrics_file = os.path.join(self.output_dir, "training_metrics.json")
        metrics_data = {
            "train_f1_score": float(train_f1),
            "train_accuracy": float(train_accuracy),
        }

        # Add validation metrics if available
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


def main():
    """Main execution."""
    # Paths
    features_dir = "/work3/monka/SummerSchool2025/results/EfficientNetB2_FocalLossLess/extracted_features/"
    output_dir = "/work3/monka/SummerSchool2025/results/XGBoost_Fusion_DefaultParams/"

    print("🍄 XGBoost Full Fusion - Kaggle Style")
    print("=" * 50)

    # Initialize
    trainer = SimpleXGBoostFusion(output_dir)

    # Train
    results = trainer.train(features_dir)  # Debug with 500 samples

    # Predict test set
    test_results = trainer.predict_test(features_dir)

    # Update saved metrics with test results
    trainer.save_model(
        pd.DataFrame(),
        results["train_f1_score"],
        results["train_accuracy"],
        results["val_f1_score"],
        results["val_accuracy"],
    )

    print("\n🏆 Training Complete!")
    print(f"Final Train F1 Score: {results['train_f1_score']:.4f}")
    print(f"Final Train Accuracy: {results['train_accuracy']:.4f}")
    print(f"Final Validation F1 Score: {results['val_f1_score']:.4f}")
    print(f"Final Validation Accuracy: {results['val_accuracy']:.4f}")
    print("Ready for submission! 🚀")


if __name__ == "__main__":
    main()
