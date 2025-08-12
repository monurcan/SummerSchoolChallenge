import os
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, accuracy_score
from xgboost import XGBClassifier
import pickle
import warnings

warnings.filterwarnings("ignore")


class MetadataFeatureEngineer:
    """
    Feature engineering for metadata to create rich features for XGBoost.
    """

    def __init__(self):
        self.habitat_encoder = TfidfVectorizer(
            max_features=50, stop_words="english", ngram_range=(1, 2)
        )
        self.substrate_encoder = TfidfVectorizer(
            max_features=30, stop_words="english", ngram_range=(1, 2)
        )
        self.habitat_label_encoder = LabelEncoder()
        self.substrate_label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.fitted = False

    def create_temporal_features(self, df):
        """Create time-based features from eventDate."""
        temporal_features = pd.DataFrame(index=df.index)

        # Parse eventDate
        df["eventDate_parsed"] = pd.to_datetime(df["eventDate"], errors="coerce")

        # Extract temporal components
        temporal_features["month"] = df["eventDate_parsed"].dt.month
        temporal_features["day_of_year"] = df["eventDate_parsed"].dt.dayofyear
        temporal_features["year"] = df["eventDate_parsed"].dt.year

        # Season mapping (Denmark seasons)
        def get_season(month):
            if pd.isna(month):
                return -1
            if month in [12, 1, 2]:
                return 0  # Winter
            elif month in [3, 4, 5]:
                return 1  # Spring
            elif month in [6, 7, 8]:
                return 2  # Summer
            else:
                return 3  # Autumn

        temporal_features["season"] = temporal_features["month"].apply(get_season)

        # Cyclic encoding for month (captures seasonal periodicity)
        temporal_features["month_sin"] = np.sin(
            2 * np.pi * temporal_features["month"] / 12
        )
        temporal_features["month_cos"] = np.cos(
            2 * np.pi * temporal_features["month"] / 12
        )

        # Fill missing values
        temporal_features = temporal_features.fillna(-1)

        return temporal_features

    def create_geographic_features(self, df):
        """Create geographic features from latitude and longitude."""
        geo_features = pd.DataFrame(index=df.index)

        # Basic coordinates
        geo_features["Latitude"] = df["Latitude"].fillna(df["Latitude"].median())
        geo_features["Longitude"] = df["Longitude"].fillna(df["Longitude"].median())

        # Denmark-specific geographic insights
        # Denmark approximate bounds: Lat 54.5-57.8, Long 8.0-15.2
        denmark_center_lat = 56.0
        denmark_center_lon = 10.0

        # Distance from geographic center of Denmark
        geo_features["dist_from_center"] = np.sqrt(
            (geo_features["Latitude"] - denmark_center_lat) ** 2
            + (geo_features["Longitude"] - denmark_center_lon) ** 2
        )

        # Coastal proximity (approximate - Denmark is very coastal)
        # Simple heuristic: closer to edges = more coastal
        geo_features["coastal_score"] = np.minimum(
            np.abs(geo_features["Latitude"] - 54.5),
            np.abs(geo_features["Latitude"] - 57.8),
        ) + np.minimum(
            np.abs(geo_features["Longitude"] - 8.0),
            np.abs(geo_features["Longitude"] - 15.2),
        )

        # Regional clusters (simple geographic zones)
        # North/South Denmark
        geo_features["north_south"] = (geo_features["Latitude"] > 56.0).astype(int)

        # East/West Denmark
        geo_features["east_west"] = (geo_features["Longitude"] > 10.5).astype(int)

        return geo_features

    def create_text_features(self, df, fit=True):
        """Create features from text fields (Habitat, Substrate)."""
        text_features = pd.DataFrame(index=df.index)

        # Fill missing text
        habitat_text = df["Habitat"].fillna("unknown").astype(str)
        substrate_text = df["Substrate"].fillna("unknown").astype(str)

        if fit:
            # Fit TF-IDF encoders
            habitat_tfidf = self.habitat_encoder.fit_transform(habitat_text)
            substrate_tfidf = self.substrate_encoder.fit_transform(substrate_text)

            # Fit label encoders
            self.habitat_label_encoder.fit(habitat_text)
            self.substrate_label_encoder.fit(substrate_text)
        else:
            # Transform using fitted encoders
            habitat_tfidf = self.habitat_encoder.transform(habitat_text)
            substrate_tfidf = self.substrate_encoder.transform(substrate_text)

        # Add TF-IDF features
        habitat_cols = [f"habitat_tfidf_{i}" for i in range(habitat_tfidf.shape[1])]
        substrate_cols = [
            f"substrate_tfidf_{i}" for i in range(substrate_tfidf.shape[1])
        ]

        habitat_df = pd.DataFrame(
            habitat_tfidf.toarray(), columns=habitat_cols, index=df.index
        )
        substrate_df = pd.DataFrame(
            substrate_tfidf.toarray(), columns=substrate_cols, index=df.index
        )

        text_features = pd.concat([text_features, habitat_df, substrate_df], axis=1)

        # Add label-encoded versions
        text_features["habitat_encoded"] = self.habitat_label_encoder.transform(
            habitat_text
        )
        text_features["substrate_encoded"] = self.substrate_label_encoder.transform(
            substrate_text
        )

        # Text length features
        text_features["habitat_length"] = habitat_text.str.len()
        text_features["substrate_length"] = substrate_text.str.len()

        # Key habitat indicators (domain knowledge)
        habitat_keywords = {
            "woodland_indicator": ["woodland", "forest", "tree"],
            "grassland_indicator": ["grass", "meadow", "field"],
            "wetland_indicator": ["bog", "marsh", "wet"],
            "urban_indicator": ["garden", "lawn", "park"],
            "deciduous_indicator": ["deciduous", "beech", "oak"],
            "coniferous_indicator": ["coniferous", "pine", "spruce"],
        }

        for feature_name, keywords in habitat_keywords.items():
            text_features[feature_name] = (
                habitat_text.str.lower()
                .str.contains("|".join(keywords), na=False)
                .astype(int)
            )

        # Key substrate indicators
        substrate_keywords = {
            "wood_substrate": ["wood", "bark", "trunk", "branch"],
            "soil_substrate": ["soil", "ground", "earth"],
            "organic_substrate": ["leaf", "litter", "compost", "mulch"],
            "living_substrate": ["living", "tree", "plant"],
        }

        for feature_name, keywords in substrate_keywords.items():
            text_features[feature_name] = (
                substrate_text.str.lower()
                .str.contains("|".join(keywords), na=False)
                .astype(int)
            )

        return text_features

    def fit_transform(self, df):
        """Fit the feature engineer and transform data."""
        temporal_features = self.create_temporal_features(df)
        geo_features = self.create_geographic_features(df)
        text_features = self.create_text_features(df, fit=True)

        # Combine all features
        all_features = pd.concat(
            [temporal_features, geo_features, text_features], axis=1
        )

        # Scale numerical features
        numerical_cols = [
            "Latitude",
            "Longitude",
            "dist_from_center",
            "coastal_score",
            "habitat_length",
            "substrate_length",
        ]
        all_features[numerical_cols] = self.scaler.fit_transform(
            all_features[numerical_cols]
        )

        self.fitted = True
        return all_features

    def transform(self, df):
        """Transform data using fitted encoders."""
        if not self.fitted:
            raise ValueError("FeatureEngineer must be fitted before transform")

        temporal_features = self.create_temporal_features(df)
        geo_features = self.create_geographic_features(df)
        text_features = self.create_text_features(df, fit=False)

        # Combine all features
        all_features = pd.concat(
            [temporal_features, geo_features, text_features], axis=1
        )

        # Scale numerical features
        numerical_cols = [
            "Latitude",
            "Longitude",
            "dist_from_center",
            "coastal_score",
            "habitat_length",
            "substrate_length",
        ]
        all_features[numerical_cols] = self.scaler.transform(
            all_features[numerical_cols]
        )

        return all_features


class XGBoostFusionTrainer:
    """
    XGBoost trainer for multimodal fungi classification.
    """

    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.feature_engineer = MetadataFeatureEngineer()
        self.models = {}
        self.feature_importance = {}
        self.cv_scores = {}

    def load_unified_dataset(self, features_dir):
        """Load the unified dataset created by feature extraction."""
        # Load main dataset
        dataset_file = os.path.join(features_dir, "unified_xgboost_dataset.parquet")
        df = pd.read_parquet(dataset_file)

        # Load feature names
        with open(os.path.join(features_dir, "feature_names.json"), "r") as f:
            feature_names = json.load(f)

        return df, feature_names

    def prepare_features(
        self,
        df,
        feature_names,
        use_image_features=True,
        use_metadata=True,
        use_probabilities=True,
    ):
        """
        Prepare feature matrix for training.
        """
        feature_columns = []

        if use_image_features:
            feature_columns.extend(feature_names["image_features"])
            print(f"Using {len(feature_names['image_features'])} image features")

        if use_probabilities:
            feature_columns.extend(feature_names["probability_features"])
            feature_columns.extend(feature_names["derived_features"])
            print(
                f"Using {len(feature_names['probability_features'])} probability features"
            )
            print(f"Using {len(feature_names['derived_features'])} derived features")

        if use_metadata:
            # Engineer metadata features
            print("Engineering metadata features...")
            if not hasattr(self, "metadata_features_fitted"):
                metadata_features = self.feature_engineer.fit_transform(df)
                self.metadata_features_fitted = True
            else:
                metadata_features = self.feature_engineer.transform(df)

            # Add metadata features to the dataframe
            for col in metadata_features.columns:
                df[f"meta_{col}"] = metadata_features[col].values
                feature_columns.append(f"meta_{col}")

            print(f"Using {len(metadata_features.columns)} metadata features")

        return df[feature_columns]

    def train_models(self, features_dir, experiment_configs=None):
        """
        Train multiple XGBoost models with different feature combinations.
        """
        # Load data
        df, feature_names = self.load_unified_dataset(features_dir)
        train_df = df[df["dataset"] == "train"].copy()

        print(f"Training data: {len(train_df)} samples")
        print(f"Number of classes: {train_df['true_label'].nunique()}")

        # Default experiment configurations
        if experiment_configs is None:
            experiment_configs = {
                "image_only": {
                    "use_image_features": True,
                    "use_metadata": False,
                    "use_probabilities": False,
                },
                "probabilities_only": {
                    "use_image_features": False,
                    "use_metadata": False,
                    "use_probabilities": True,
                },
                "metadata_only": {
                    "use_image_features": False,
                    "use_metadata": True,
                    "use_probabilities": False,
                },
                "image_plus_metadata": {
                    "use_image_features": True,
                    "use_metadata": True,
                    "use_probabilities": False,
                },
                "probabilities_plus_metadata": {
                    "use_image_features": False,
                    "use_metadata": True,
                    "use_probabilities": True,
                },
                "full_fusion": {
                    "use_image_features": True,
                    "use_metadata": True,
                    "use_probabilities": True,
                },
            }

        # XGBoost parameters
        xgb_params = {
            "objective": "multi:softprob",
            "num_class": 183,
            "max_depth": 8,
            "learning_rate": 0.1,
            "n_estimators": 1000,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "n_jobs": -1,
            "early_stopping_rounds": 50,
            "eval_metric": "mlogloss",
        }

        results = {}

        for exp_name, config in experiment_configs.items():
            print(f"\n{'=' * 60}")
            print(f"Training experiment: {exp_name}")
            print(f"{'=' * 60}")

            # Prepare features
            X = self.prepare_features(train_df, feature_names, **config)
            y = train_df["true_label"].values

            print(f"Feature matrix shape: {X.shape}")
            print(f"Missing values: {X.isnull().sum().sum()}")

            # Handle missing values
            X = X.fillna(-999)  # XGBoost handles missing values, but explicit is better

            # Cross-validation
            cv_scores = self.cross_validate_model(X, y, xgb_params)

            # Train final model
            model = XGBClassifier(**xgb_params)
            model.fit(X, y, verbose=False)

            # Store results
            self.models[exp_name] = model
            self.cv_scores[exp_name] = cv_scores

            # Feature importance
            importance_df = pd.DataFrame(
                {"feature": X.columns, "importance": model.feature_importances_}
            ).sort_values("importance", ascending=False)

            self.feature_importance[exp_name] = importance_df

            results[exp_name] = {
                "cv_scores": cv_scores,
                "feature_count": X.shape[1],
                "top_features": importance_df.head(10)["feature"].tolist(),
            }

            print(f"CV F1 Score: {cv_scores['f1']:.4f} (+/- {cv_scores['f1_std']:.4f})")
            print(
                f"CV Accuracy: {cv_scores['accuracy']:.4f} (+/- {cv_scores['accuracy_std']:.4f})"
            )

        # Save models and results
        self.save_results(results)

        return results

    def cross_validate_model(self, X, y, xgb_params, cv_folds=5):
        """Perform cross-validation."""
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        f1_scores = []
        accuracy_scores = []

        for train_idx, val_idx in skf.split(X, y):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]

            # Train model
            model = XGBClassifier(**xgb_params)
            model.fit(
                X_train_fold,
                y_train_fold,
                eval_set=[(X_val_fold, y_val_fold)],
                verbose=False,
            )

            # Predict
            y_pred = model.predict(X_val_fold)

            # Calculate metrics
            f1_scores.append(f1_score(y_val_fold, y_pred, average="weighted"))
            accuracy_scores.append(accuracy_score(y_val_fold, y_pred))

        return {
            "f1": np.mean(f1_scores),
            "f1_std": np.std(f1_scores),
            "accuracy": np.mean(accuracy_scores),
            "accuracy_std": np.std(accuracy_scores),
        }

    def predict_test_set(self, features_dir, model_name="full_fusion"):
        """Generate predictions for test set."""
        df, feature_names = self.load_unified_dataset(features_dir)
        test_df = df[df["dataset"] == "test"].copy()

        if model_name not in self.models:
            raise ValueError(
                f"Model {model_name} not found. Available: {list(self.models.keys())}"
            )

        # Get the same feature configuration used for training
        if model_name == "full_fusion":
            config = {
                "use_image_features": True,
                "use_metadata": True,
                "use_probabilities": True,
            }
        elif model_name == "image_plus_metadata":
            config = {
                "use_image_features": True,
                "use_metadata": True,
                "use_probabilities": False,
            }
        else:
            # Add other configurations as needed
            config = {
                "use_image_features": True,
                "use_metadata": True,
                "use_probabilities": True,
            }

        X_test = self.prepare_features(test_df, feature_names, **config)
        X_test = X_test.fillna(-999)

        # Predict
        model = self.models[model_name]
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)

        # Create results DataFrame
        results_df = pd.DataFrame(
            {
                "filename": test_df["filename"].values,
                "predicted_label": predictions,
                "prediction_confidence": np.max(probabilities, axis=1),
            }
        )

        # Save results
        output_file = os.path.join(
            self.output_dir, f"test_predictions_{model_name}.csv"
        )
        with open(output_file, "w") as f:
            f.write(f"XGBoost_Fusion_{model_name}\n")
            for _, row in results_df.iterrows():
                f.write(f"{row['filename']},{row['predicted_label']}\n")

        print(f"Test predictions saved to {output_file}")
        return results_df

    def save_results(self, results):
        """Save models and results."""
        os.makedirs(self.output_dir, exist_ok=True)

        # Save models
        for name, model in self.models.items():
            model_file = os.path.join(self.output_dir, f"model_{name}.pkl")
            with open(model_file, "wb") as f:
                pickle.dump(model, f)

        # Save feature engineer
        fe_file = os.path.join(self.output_dir, "feature_engineer.pkl")
        with open(fe_file, "wb") as f:
            pickle.dump(self.feature_engineer, f)

        # Save results summary
        results_file = os.path.join(self.output_dir, "training_results.json")
        with open(results_file, "w") as f:
            # Convert numpy types to native Python types for JSON serialization
            json_results = {}
            for exp_name, exp_results in results.items():
                json_results[exp_name] = {
                    "cv_f1": float(exp_results["cv_scores"]["f1"]),
                    "cv_f1_std": float(exp_results["cv_scores"]["f1_std"]),
                    "cv_accuracy": float(exp_results["cv_scores"]["accuracy"]),
                    "cv_accuracy_std": float(exp_results["cv_scores"]["accuracy_std"]),
                    "feature_count": int(exp_results["feature_count"]),
                    "top_features": exp_results["top_features"],
                }
            json.dump(json_results, f, indent=2)

        # Save feature importance
        for name, importance_df in self.feature_importance.items():
            importance_file = os.path.join(
                self.output_dir, f"feature_importance_{name}.csv"
            )
            importance_df.to_csv(importance_file, index=False)

        print(f"Results saved to {self.output_dir}")


def main():
    """Main training script."""
    # Configuration
    features_dir = "/work3/monka/SummerSchool2025/results/EfficientNetB2_FocalLossLess/extracted_features/"
    output_dir = "/work3/monka/SummerSchool2025/results/XGBoost_Fusion/"

    # Initialize trainer
    trainer = XGBoostFusionTrainer(output_dir)

    # Train models
    print("Starting XGBoost fusion training...")
    results = trainer.train_models(features_dir)

    # Print summary
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)

    best_model = None
    best_score = 0

    for exp_name, exp_results in results.items():
        f1_score = exp_results["cv_scores"]["f1"]
        print(
            f"{exp_name:25} | F1: {f1_score:.4f} | Features: {exp_results['feature_count']:4d}"
        )

        if f1_score > best_score:
            best_score = f1_score
            best_model = exp_name

    print(f"\nBest model: {best_model} (F1: {best_score:.4f})")

    # Generate test predictions with best model
    print(f"\nGenerating test predictions with {best_model}...")
    trainer.predict_test_set(features_dir, best_model)

    print("\nTraining completed! 🍄")
    print(f"Check results in: {output_dir}")


if __name__ == "__main__":
    main()
