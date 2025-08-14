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
            # New options for balancing image vs metadata features
            "reduce_image_dimensions": True,
            "target_image_dimensions": 100,  # Reduce to this many dimensions
            # Hyperparameter optimization
            "do_hyperparamopt": False,
            "hyperparamopt_max_evals": 50,
        }

        # Update with user config
        if feature_config:
            self.feature_config.update(feature_config)

        self.print_config()

    def print_config(self):
        """Print current feature configuration."""
        print("\n🔧 Feature Configuration:")
        print(f"  ✅ Image features: {self.feature_config['use_image_features']}")
        if self.feature_config["use_image_features"]:
            print(
                f"    🔧 Reduce dimensions: {self.feature_config.get('reduce_image_dimensions', False)}"
            )
            if self.feature_config.get("reduce_image_dimensions", False):
                print(
                    f"    📏 Target dimensions: {self.feature_config.get('target_image_dimensions', 100)}"
                )
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
        print(
            f"  🔍 Hyperparameter optimization: {self.feature_config['do_hyperparamopt']}"
        )
        if self.feature_config.get("do_hyperparamopt", False):
            print(
                f"  🎯 Max evaluations: {self.feature_config['hyperparamopt_max_evals']}"
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

        # 1. Image features (toggleable) - with dimensionality balancing
        if self.feature_config["use_image_features"]:
            print("  📸 Adding image features...")
            image_features = np.vstack(df["image_features_vector"].values)

            # Option 1: Dimensionality reduction to balance with metadata features
            reduce_image_dims = self.feature_config.get("reduce_image_dimensions", True)
            target_image_dims = self.feature_config.get("target_image_dimensions", 100)

            if reduce_image_dims and image_features.shape[1] > target_image_dims:
                print(
                    f"  🔧 Reducing image features from {image_features.shape[1]} to {target_image_dims} dimensions"
                )
                from sklearn.decomposition import PCA

                if is_train:
                    # Fit PCA on training data
                    self.image_pca = PCA(
                        n_components=target_image_dims, random_state=42
                    )
                    image_features_reduced = self.image_pca.fit_transform(
                        image_features
                    )
                else:
                    # Transform test data using fitted PCA
                    if hasattr(self, "image_pca"):
                        image_features_reduced = self.image_pca.transform(
                            image_features
                        )
                    else:
                        raise ValueError("PCA model not fitted. Train the model first.")

                image_features = image_features_reduced
                print(f"  ✅ Image features reduced to shape: {image_features.shape}")

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

    def define_hyperparameter_space(self, num_classes):
        """Define the hyperparameter search space for XGBoost."""
        from hyperopt import hp

        space = {
            "max_depth": hp.quniform("max_depth", 3, 18, 1),
            "gamma": hp.uniform("gamma", 1, 9),
            "reg_alpha": hp.quniform("reg_alpha", 40, 180, 1),
            "reg_lambda": hp.uniform("reg_lambda", 0, 1),
            "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1),
            "min_child_weight": hp.quniform("min_child_weight", 0, 10, 1),
            "n_estimators": 180,
            "learning_rate": hp.uniform("learning_rate", 0.01, 0.3),
            "subsample": hp.uniform("subsample", 0.6, 1.0),
            "seed": 0,
            # Fixed parameters
            "objective": "multi:softprob",
            "num_class": num_classes,
            "random_state": 42,
            "tree_method": "gpu_hist",
            "enable_categorical": True,
            "device": "cuda",
            "verbosity": 0,  # Reduce verbosity during hyperopt
        }

        return space

    def hyperopt_objective(self, params, X_train, y_train, X_val, y_val):
        """Objective function for hyperparameter optimization."""
        from hyperopt import STATUS_OK

        # Convert discrete parameters to int
        params["max_depth"] = int(params["max_depth"])
        params["reg_alpha"] = int(params["reg_alpha"])
        params["min_child_weight"] = int(params["min_child_weight"])
        params["n_estimators"] = int(params["n_estimators"])

        try:
            # Create and train model with current parameters
            model = XGBClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

            # Get validation predictions
            val_predictions = model.predict(X_val)

            # Calculate F1 score (we want to maximize this, so return negative)
            f1 = f1_score(y_val, val_predictions, average="weighted")

            # Return negative F1 because hyperopt minimizes
            return {"loss": -f1, "status": STATUS_OK}

        except Exception as e:
            print(f"Error in hyperopt objective: {e}")
            # Return a large loss if there's an error
            return {"loss": 1.0, "status": STATUS_OK}

    def optimize_hyperparameters(self, X_train, y_train, X_val, y_val):
        """Perform hyperparameter optimization using Hyperopt."""
        from hyperopt import fmin, tpe, Trials

        print("🔍 Starting hyperparameter optimization...")

        # Define search space
        space = self.define_hyperparameter_space(len(np.unique(y_train)))

        # Create trials object to store results
        trials = Trials()

        # Define objective function with data
        def objective(params):
            return self.hyperopt_objective(params, X_train, y_train, X_val, y_val)

        # Run optimization
        max_evals = self.feature_config.get("hyperparamopt_max_evals", 50)

        print(f"  🎯 Running {max_evals} evaluations...")
        best = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials,
            verbose=True,
        )

        # Convert best parameters back to correct types
        best["max_depth"] = int(best["max_depth"])
        best["reg_alpha"] = int(best["reg_alpha"])
        best["min_child_weight"] = int(best["min_child_weight"])
        best["n_estimators"] = int(best.get("n_estimators", 180))

        # Add fixed parameters
        best.update(
            {
                "objective": "multi:softprob",
                "num_class": len(np.unique(y_train)),
                "random_state": 42,
                "tree_method": "gpu_hist",
                "enable_categorical": True,
                "device": "cuda",
                "verbosity": 1,
                "seed": 0,
            }
        )

        # Get best F1 score
        best_f1 = -min(trials.losses())

        print(f"  ✅ Best validation F1 score: {best_f1:.4f}")
        print("  🏆 Best parameters:")
        for key, value in best.items():
            if key not in [
                "objective",
                "num_class",
                "random_state",
                "tree_method",
                "enable_categorical",
                "device",
                "verbosity",
                "seed",
            ]:
                print(f"    {key}: {value}")

        return best, best_f1, trials

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
                    "use_image_features": self.feature_config["use_image_features"],
                    "use_class_probabilities": self.feature_config[
                        "use_class_probabilities"
                    ],
                    "use_prediction_confidence": self.feature_config[
                        "use_prediction_confidence"
                    ],
                    "use_prediction_entropy": self.feature_config[
                        "use_prediction_entropy"
                    ],
                    "use_metadata_features": self.feature_config[
                        "use_metadata_features"
                    ],
                }
            )

        # Split into train and validation sets
        print("Splitting data into train/validation...")
        train_split, val_split = train_test_split(
            train_df, test_size=0.015, random_state=42, stratify=train_df["true_label"]
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

        # XGBoost parameters - use hyperparameter optimization if enabled
        if self.feature_config.get("do_hyperparamopt", False):
            print("🔍 Hyperparameter optimization enabled - finding best parameters...")
            best_params, best_f1, trials = self.optimize_hyperparameters(
                X_train, y_train_encoded, X_val, y_val_encoded
            )
            params = best_params

            # Save hyperopt results
            hyperopt_results = {
                "best_params": best_params,
                "best_validation_f1": best_f1,
                "trials_count": len(trials.trials),
            }

            # Store hyperopt results for saving later
            self.hyperopt_results = hyperopt_results

        else:
            # Default parameters
            params = {
                "objective": "multi:softprob",
                "num_class": len(np.unique(y_train_encoded)),
                "random_state": 42,
                "tree_method": "gpu_hist",
                "enable_categorical": True,
                "device": "cuda",
                "verbosity": 1,
                # "colsample_bytree": 0.9874126599523046,
                # "gamma": 4.172609851489677,
                # "learning_rate": 0.13331391004782508,
                # "max_depth": 5,
                # "min_child_weight": 2,
                # "reg_alpha": 84,
                # "reg_lambda": 0.6672388027430995,
                # "subsample": 0.818384193819345,
                # "n_estimators": 180,
                # Feature sampling to prevent image features from dominating
                # "colsample_bytree": 0.3,  # Sample fewer features per tree to give metadata a chance
                # "colsample_bylevel": 0.7,  # Additional feature sampling per level
                # "colsample_bynode": 0.8,  # Additional feature sampling per node
                # # Regularization to prevent overfitting on high-dim image features
                # "reg_alpha": 10.0,  # L1 regularization
                # "reg_lambda": 10.0,  # L2 regularization
                # "max_depth": 6,  # Shallower trees to prevent overfitting
                # "min_child_weight": 3,  # Higher minimum samples per leaf
                # "learning_rate": 0.1,  # Lower learning rate for better generalization
                # "n_estimators": 1000,  # More trees with lower learning rate
                # "subsample": 0.8,  # Row subsampling
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
        test_df = df[(df["dataset"] == "test") | (df["dataset"] == "final")].copy()

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

                use_tta_predictions_from_csv = True
                if use_tta_predictions_from_csv:
                    # Instead of using the class probabilities, load TTA predictions from CSV
                    tta_csv_path = "/work3/monka/SummerSchool2025/results/EfficientNet_V2L_CrossEntropy_New/test_probabilities_tta_64.csv"
                    print(f"  🔄 Loading TTA predictions from: {tta_csv_path}")

                    # Load TTA predictions CSV
                    tta_df = pd.read_csv(tta_csv_path)

                    # Get test filenames in the current order
                    test_filenames = test_df["filename"].values

                    # Create mapping from filename to TTA predictions
                    # First column is filename, rest are class probabilities
                    tta_filename_col = tta_df.columns[
                        0
                    ]  # First column contains filenames
                    tta_prob_cols = tta_df.columns[1:]  # Rest are probability columns

                    # Create a mapping from filename to probabilities
                    tta_dict = {}
                    for idx, row in tta_df.iterrows():
                        filename = row[tta_filename_col]
                        probabilities = row[tta_prob_cols].values
                        tta_dict[filename] = probabilities

                    # Order TTA predictions to match test data order
                    tta_predictions_ordered = []
                    missing_files = []

                    for filename in test_filenames:
                        if filename in tta_dict:
                            tta_predictions_ordered.append(tta_dict[filename])
                        else:
                            missing_files.append(filename)
                            # Use zeros as fallback if file not found in TTA predictions
                            tta_predictions_ordered.append(np.zeros(len(tta_prob_cols)))

                    if missing_files:
                        print(
                            f"  ⚠️  Warning: {len(missing_files)} files not found in TTA predictions, using zeros as fallback"
                        )
                        print(f"  First few missing: {missing_files[:5]}")

                    # Convert to numpy array and replace image_probabilities
                    image_probabilities = np.array(tta_predictions_ordered)
                    print(
                        f"  ✅ Loaded TTA predictions shape: {image_probabilities.shape}"
                    )

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
        if config.get("do_hyperparamopt", False):
            suffix_parts.append("HYPEROPT")

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
        model_data = {
            "model": self.model,
            "label_encoder": self.label_encoder,
            "feature_config": self.feature_config,
            "categorical_mappings": self.categorical_mappings,
        }

        # Save PCA model if it exists
        if hasattr(self, "image_pca"):
            model_data["image_pca"] = self.image_pca

        # Save hyperparameter optimization results if they exist
        if hasattr(self, "hyperopt_results"):
            model_data["hyperopt_results"] = self.hyperopt_results

        with open(model_file, "wb") as f:
            pickle.dump(model_data, f)

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

        # Add hyperparameter optimization results if available
        if hasattr(self, "hyperopt_results"):
            metrics_data["hyperopt_results"] = self.hyperopt_results

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
    output_dir = "/work3/monka/SummerSchool2025/results/XGBoost_Configurable_Experiments_WithUpdatedMetadata_15/"

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
        # {
        #     "name": "Only Metadata",
        #     "config": {
        #         "use_image_features": False,
        #         "use_class_probabilities": False,
        #         "use_prediction_confidence": False,
        #         "use_prediction_entropy": False,
        #         "use_metadata_features": True,
        #     },
        # },
        # {
        #     "name": "Metadata Multiply Image Fusion 1.0",
        #     "config": {
        #         "use_image_features": False,  # Will be forced in the fusion strategy
        #         "use_class_probabilities": False,
        #         "use_prediction_confidence": False,
        #         "use_prediction_entropy": False,
        #         "use_metadata_features": True,
        #         "metadata_multiply_image_fusion": True,
        #         "metadata_fusion_temperature": 1.0,  # Configurable temperature for smoothing
        #     },
        # },
        {
            "name": "Metadata Multiply Image Fusion 1.5",
            "config": {
                "use_image_features": False,  # Will be forced in the fusion strategy
                "use_class_probabilities": False,
                "use_prediction_confidence": False,
                "use_prediction_entropy": False,
                "use_metadata_features": True,
                "metadata_multiply_image_fusion": True,
                "metadata_fusion_temperature": 1.5,  # Configurable temperature for smoothing
            },
        },
        # {
        #     "name": "Metadata Multiply Image Fusion 1.75",
        #     "config": {
        #         "use_image_features": False,  # Will be forced in the fusion strategy
        #         "use_class_probabilities": False,
        #         "use_prediction_confidence": False,
        #         "use_prediction_entropy": False,
        #         "use_metadata_features": True,
        #         "metadata_multiply_image_fusion": True,
        #         "metadata_fusion_temperature": 1.75,  # Configurable temperature for smoothing
        #     },
        # },
        # {
        #     "name": "Metadata Multiply Image Fusion 2.0",
        #     "config": {
        #         "use_image_features": False,
        #         "use_class_probabilities": False,
        #         "use_prediction_confidence": False,
        #         "use_prediction_entropy": False,
        #         "use_metadata_features": True,
        #         "metadata_multiply_image_fusion": True,
        #         "metadata_fusion_temperature": 2.0,  # Configurable temperature for smoothing
        #     },
        # },
        # {
        #     "name": "Metadata Multiply Image Fusion 2.5",
        #     "config": {
        #         "use_image_features": False,  # Will be forced in the fusion strategy
        #         "use_class_probabilities": False,
        #         "use_prediction_confidence": False,
        #         "use_prediction_entropy": False,
        #         "use_metadata_features": True,
        #         "metadata_multiply_image_fusion": True,
        #         "metadata_fusion_temperature": 2.5,  # Configurable temperature for smoothing
        #     },
        # },
        # {
        #     "name": "Metadata with HyperOpt",
        #     "config": {
        #         "use_image_features": False,
        #         "use_class_probabilities": False,
        #         "use_prediction_confidence": False,
        #         "use_prediction_entropy": False,
        #         "use_metadata_features": True,
        #         "metadata_multiply_image_fusion": False,
        #         "do_hyperparamopt": True,
        #         "hyperparamopt_max_evals": 40,  # Reduced for faster execution
        #     },
        # },
        # {
        #     "name": "Metadata Multiply Image Fusion 2.0 with Class Probs",
        #     "config": {
        #         "use_image_features": False,  # Will be forced in the fusion strategy
        #         "use_class_probabilities": True,
        #         "use_prediction_confidence": True,
        #         "use_prediction_entropy": True,
        #         "use_metadata_features": True,
        #         "metadata_multiply_image_fusion": True,
        #         "metadata_fusion_temperature": 2.0,  # Configurable temperature for smoothing
        #     },
        # },
        # {
        #     "name": "Metadata Multiply Image Fusion 2.0 with Img Features",
        #     "config": {
        #         "use_image_features": True,  # Will be forced in the fusion strategy
        #         "use_class_probabilities": False,
        #         "use_prediction_confidence": False,
        #         "use_prediction_entropy": False,
        #         "use_metadata_features": True,
        #         "metadata_multiply_image_fusion": True,
        #         "metadata_fusion_temperature": 2.0,  # Configurable temperature for smoothing
        #         "reduce_image_dimensions": True,
        #         "target_image_dimensions": 10,
        #     },
        # },
        # {
        #     "name": "Metadat + Compressed Image",
        #     "config": {
        #         "use_image_features": True,  # Will be forced in the fusion strategy
        #         "use_class_probabilities": False,
        #         "use_prediction_confidence": False,
        #         "use_prediction_entropy": False,
        #         "use_metadata_features": True,
        #         "metadata_multiply_image_fusion": False,
        #         "reduce_image_dimensions": True,
        #         "target_image_dimensions": 10,
        #     },
        # },
        # {
        #     "name": "Metadata Multiply Image Fusion 4.0",
        #     "config": {
        #         "use_image_features": False,  # Will be forced in the fusion strategy
        #         "use_class_probabilities": False,
        #         "use_prediction_confidence": False,
        #         "use_prediction_entropy": False,
        #         "use_metadata_features": True,
        #         "metadata_multiply_image_fusion": True,
        #         "metadata_fusion_temperature": 4.0,  # Configurable temperature for smoothing
        #     },
        # },
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
