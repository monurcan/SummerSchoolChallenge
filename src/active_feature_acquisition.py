#!/usr/bin/env python3
"""
Active Feature Acquisition for Fungi Challenge

This script implements an active learning strategy for selecting the most informative
training samples to acquire metadata for, based on training loss and model uncertainty.

The strategy prioritizes samples with:
1. High training loss (samples the model struggles with most)
2. High prediction uncertainty (entropy of softmax outputs)
3. Diverse class predictions (to cover different failure modes)

KEY IMPROVEMENT: Uses 3-fold cross-validation models for better loss estimation.
For each sample, the script uses the model that did NOT see that sample during training
(i.e., the model where that sample was in the validation set). This provides a much more
reliable estimate of sample difficulty compared to using a single model trained on all data.

FEATURE SELECTION STRATEGIES:
- Greedy: Selects all missing features for a sample in priority order
- Probabilistic (default): Randomly samples features based on their importance probabilities

Usage:
    python active_feature_acquisition.py --budget 1000 --results_dir /path/to/3fold_results --strategy loss --feature_strategy probabilistic
"""

import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import csv
import argparse
import random
from collections import Counter
from torchvision import models
from fungi_network_b2_3fold_cv_model_training import (
    FungiDataset,
    get_transforms,
    load_fold_indices,
)


class ActiveFeatureSelector:
    def __init__(
        self, metadata_file, image_path, results_dir, device="cuda", use_3fold_cv=True
    ):
        """
        Initialize the Active Feature Selector

        Args:
            metadata_file: Path to metadata CSV file
            image_path: Path to fungi images directory
            results_dir: Path to results directory containing trained models
            device: Device to run inference on
            use_3fold_cv: Whether to use 3-fold CV models for better loss estimation
        """
        self.metadata_file = metadata_file
        self.image_path = image_path
        self.results_dir = results_dir
        self.use_3fold_cv = use_3fold_cv
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Metadata costs
        self.metadata_costs = {
            "eventDate": 2,
            "Latitude": 1,
            "Longitude": 1,
            "Habitat": 2,
            "Substrate": 2,
        }

        # Feature priority order (most informative first)
        self.feature_priority = [
            "Habitat",
            "Substrate",
            "eventDate",
            "Latitude",
            "Longitude",
        ]

        # Feature sampling probabilities
        self.feature_sampling_probabilities = {
            "Habitat": 0.75,
            "Substrate": 0.65,
            "eventDate": 0.6,
            "Latitude": 0.15,
            "Longitude": 0.15,
        }

        # Load data
        self.df = pd.read_csv(metadata_file)
        self.train_df = self.df[self.df["filename_index"].str.startswith("fungi_train")]

        if use_3fold_cv:
            # Load fold indices and models for 3-fold CV
            self.fold_indices = load_fold_indices(results_dir)
            self.models = self._load_3fold_models()
            self.sample_to_fold = self._create_sample_to_fold_mapping()
        else:
            # Load single model
            model_path = os.path.join(results_dir, "best_accuracy.pth")
            self.model = self._load_single_model(model_path)

    def _load_single_model(self, model_path):
        """Load a single trained model"""
        model = models.efficientnet_b2(pretrained=True)
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(model.classifier[1].in_features, 183),  # 183 classes
        )
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    def _load_3fold_models(self):
        """Load all three 3-fold CV models"""
        models_dict = {}
        for fold in range(1, 4):
            model_path = os.path.join(
                self.results_dir, f"best_accuracy_validation{fold}.pth"
            )
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found: {model_path}")

            model = models.efficientnet_b2(pretrained=True)
            model.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(model.classifier[1].in_features, 183),  # 183 classes
            )
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.to(self.device)
            model.eval()
            models_dict[fold] = model
            print(f"Loaded model for fold {fold}")

        return models_dict

    def _create_sample_to_fold_mapping(self):
        """Create mapping from sample filename to validation fold"""
        # Reset index to match the saved indices
        train_df_reset = self.train_df.reset_index(drop=True)

        sample_to_fold = {}

        # For each fold, map validation samples to that fold
        for fold in range(1, 4):
            val_indices = self.fold_indices[f"fold_{fold}_val_indices"]
            for idx in val_indices:
                filename = train_df_reset.iloc[idx]["filename_index"]
                sample_to_fold[filename] = fold

        print(f"Created sample-to-fold mapping for {len(sample_to_fold)} samples")
        return sample_to_fold

    def compute_sample_informativeness(self, strategy="loss"):
        """
        Compute informativeness scores for each training sample
        For 3-fold CV: uses the model that did NOT see this sample during training

        Args:
            strategy: 'loss', 'uncertainty', or 'combined'

        Returns:
            Dictionary mapping filename to informativeness score
        """
        if self.use_3fold_cv:
            return self._compute_informativeness_3fold(strategy)
        else:
            return self._compute_informativeness_single(strategy)

    def _compute_informativeness_single(self, strategy):
        """Compute informativeness using a single model"""
        train_dataset = FungiDataset(
            self.train_df, self.image_path, transform=get_transforms(data="valid")
        )
        train_loader = DataLoader(
            train_dataset, batch_size=64, shuffle=False, num_workers=4
        )

        sample_scores = {}
        criterion = nn.CrossEntropyLoss(reduction="none")

        print(f"Computing informativeness scores using strategy: {strategy}")

        with torch.no_grad():
            for images, labels, filenames in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)

                # Compute losses for each sample
                losses = criterion(outputs, labels).cpu().numpy()

                # Compute prediction uncertainty (entropy)
                probs = F.softmax(outputs, dim=1)
                entropy = (
                    -torch.sum(probs * torch.log(probs + 1e-8), dim=1).cpu().numpy()
                )

                # Compute confidence (max probability)
                confidence = torch.max(probs, dim=1)[0].cpu().numpy()
                uncertainty = 1 - confidence

                for i, filename in enumerate(filenames):
                    if strategy == "loss":
                        score = losses[i]
                    elif strategy == "uncertainty":
                        score = entropy[i]
                    elif strategy == "confidence":
                        score = uncertainty[i]
                    elif strategy == "combined":
                        # Combine loss and uncertainty (normalize first)
                        score = 0.7 * losses[i] + 0.3 * entropy[i]
                    else:
                        raise ValueError(f"Unknown strategy: {strategy}")

                    sample_scores[filename] = float(score)

        return sample_scores

    def _compute_informativeness_3fold(self, strategy):
        """Compute informativeness using 3-fold CV models"""
        sample_scores = {}
        criterion = nn.CrossEntropyLoss(reduction="none")

        print(f"Computing informativeness scores using 3-fold CV strategy: {strategy}")

        # Create a single dataset for all samples
        train_dataset = FungiDataset(
            self.train_df, self.image_path, transform=get_transforms(data="valid")
        )
        train_loader = DataLoader(
            train_dataset, batch_size=64, shuffle=False, num_workers=4
        )

        with torch.no_grad():
            for images, labels, filenames in train_loader:
                # Group current batch samples by fold
                batch_by_fold = {1: [], 2: [], 3: []}

                for i, filename in enumerate(filenames):
                    if filename in self.sample_to_fold:
                        fold = self.sample_to_fold[filename]
                        batch_by_fold[fold].append(
                            {
                                "idx": i,
                                "filename": filename,
                                "image": images[i : i + 1],
                                "label": labels[i : i + 1],
                            }
                        )

                # Process each fold's samples with the corresponding model
                for fold in range(1, 4):
                    if not batch_by_fold[fold]:
                        continue

                    model = self.models[fold]
                    fold_samples = batch_by_fold[fold]

                    # Concatenate images and labels for this fold
                    fold_images = torch.cat(
                        [s["image"] for s in fold_samples], dim=0
                    ).to(self.device)
                    fold_labels = torch.cat(
                        [s["label"] for s in fold_samples], dim=0
                    ).to(self.device)

                    # Get model outputs
                    outputs = model(fold_images)

                    # Compute losses for each sample
                    losses = criterion(outputs, fold_labels).cpu().numpy()

                    # Compute prediction uncertainty (entropy)
                    probs = F.softmax(outputs, dim=1)
                    entropy = (
                        -torch.sum(probs * torch.log(probs + 1e-8), dim=1).cpu().numpy()
                    )

                    # Compute confidence (max probability)
                    confidence = torch.max(probs, dim=1)[0].cpu().numpy()
                    uncertainty = 1 - confidence

                    # Store scores for each sample
                    for i, sample in enumerate(fold_samples):
                        filename = sample["filename"]

                        if strategy == "loss":
                            score = losses[i]
                        elif strategy == "uncertainty":
                            score = entropy[i]
                        elif strategy == "confidence":
                            score = uncertainty[i]
                        elif strategy == "combined":
                            # Combine loss and uncertainty
                            score = 0.7 * losses[i] + 0.3 * entropy[i]
                        else:
                            raise ValueError(f"Unknown strategy: {strategy}")

                        sample_scores[filename] = float(score)

        print(f"Computed scores for {len(sample_scores)} samples using 3-fold CV")
        return sample_scores

    def get_missing_metadata(self, filename):
        """Get list of missing metadata features for a given sample"""
        row = self.train_df[self.train_df["filename_index"] == filename].iloc[0]
        missing_features = []

        for feature in self.feature_priority:
            if pd.isna(row[feature]) or row[feature] == "":
                missing_features.append(feature)

        return missing_features

    def select_samples_and_features(
        self,
        budget,
        strategy="loss",
        diversity_factor=0.1,
        feature_strategy="probabilistic",
    ):
        """
        Select the most informative samples and features given a budget

        Args:
            budget: Total budget in credits
            strategy: Strategy for computing informativeness
            diversity_factor: Factor to encourage diversity in selected samples
            feature_strategy: Strategy for selecting features ('greedy' or 'probabilistic')

        Returns:
            List of (filename, feature) tuples to acquire
        """
        print(f"Selecting samples with budget: {budget} credits")
        print(f"Using feature selection strategy: {feature_strategy}")

        # Compute informativeness scores
        sample_scores = self.compute_sample_informativeness(strategy)

        # Sort samples by informativeness (highest first)
        sorted_samples = sorted(sample_scores.items(), key=lambda x: x[1], reverse=True)

        selected_acquisitions = []
        remaining_budget = budget
        selected_classes = set()

        # Track which samples we've already selected features for
        samples_processed = set()

        for filename, score in sorted_samples:
            if remaining_budget <= 0:
                break

            # Skip if we've already processed this sample
            if filename in samples_processed:
                continue

            missing_features = self.get_missing_metadata(filename)

            if not missing_features:
                continue  # Skip samples that already have all metadata

            # Get the true class for diversity
            sample_class = self.train_df[self.train_df["filename_index"] == filename][
                "taxonID_index"
            ].iloc[0]

            # Diversity bonus: prefer samples from classes we haven't selected yet
            diversity_bonus = (
                diversity_factor if sample_class not in selected_classes else 0
            )
            adjusted_score = score + diversity_bonus

            # Select features for this sample based on strategy
            if feature_strategy == "greedy":
                sample_acquisitions = self._select_features_greedy(
                    filename, missing_features, remaining_budget
                )
            elif feature_strategy == "probabilistic":
                sample_acquisitions = self._select_features_probabilistic(
                    filename, missing_features, remaining_budget
                )
            else:
                raise ValueError(f"Unknown feature strategy: {feature_strategy}")

            # If we can acquire at least one feature for this sample, add it
            if sample_acquisitions:
                sample_cost = sum(
                    self.metadata_costs[feature] for _, feature in sample_acquisitions
                )
                selected_acquisitions.extend(sample_acquisitions)
                remaining_budget -= sample_cost
                selected_classes.add(sample_class)
                samples_processed.add(filename)

                print(
                    f"Selected {filename} (score: {adjusted_score:.4f}, class: {sample_class}) "
                    f"- {len(sample_acquisitions)} features, cost: {sample_cost}"
                )

        total_cost = sum(
            self.metadata_costs[feature] for _, feature in selected_acquisitions
        )
        print(f"\nTotal selected: {len(selected_acquisitions)} acquisitions")
        print(f"Total cost: {total_cost} credits")
        print(f"Remaining budget: {budget - total_cost} credits")

        return selected_acquisitions

    def _select_features_greedy(self, filename, missing_features, remaining_budget):
        """
        Greedy feature selection: select features in priority order until budget is exhausted
        """
        sample_acquisitions = []
        sample_cost = 0

        for feature in missing_features:
            feature_cost = self.metadata_costs[feature]
            if sample_cost + feature_cost <= remaining_budget:
                sample_acquisitions.append((filename, feature))
                sample_cost += feature_cost
            else:
                break  # Can't afford this feature, move to next sample

        return sample_acquisitions

    def _select_features_probabilistic(
        self, filename, missing_features, remaining_budget
    ):
        """
        Probabilistic feature selection: randomly sample features based on their probabilities
        """
        sample_acquisitions = []
        sample_cost = 0

        for feature in missing_features:
            feature_cost = self.metadata_costs[feature]

            # Check if we can afford this feature
            if sample_cost + feature_cost <= remaining_budget:
                # Generate random number and compare with feature probability
                if random.random() < self.feature_sampling_probabilities[feature]:
                    sample_acquisitions.append((filename, feature))
                    sample_cost += feature_cost

        return sample_acquisitions

    def create_shopping_list(self, acquisitions, output_file):
        """Create a shopping list CSV file from selected acquisitions"""
        with open(output_file, mode="w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerows(acquisitions)

        print(f"Shopping list saved to: {output_file}")

        # Print summary
        feature_counts = Counter(feature for _, feature in acquisitions)
        print("\nFeature acquisition summary:")
        total_cost = 0
        for feature, count in feature_counts.items():
            cost = count * self.metadata_costs[feature]
            total_cost += cost
            print(f"  {feature}: {count} samples, {cost} credits")
        print(f"  Total: {len(acquisitions)} acquisitions, {total_cost} credits")


def main():
    parser = argparse.ArgumentParser(
        description="Active Feature Acquisition for Fungi Challenge"
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=1000,
        help="Budget in credits for feature acquisition",
    )
    parser.add_argument(
        "--strategy",
        choices=["loss", "uncertainty", "confidence", "combined"],
        default="loss",
        help="Strategy for selecting informative samples",
    )
    parser.add_argument(
        "--feature_strategy",
        choices=["greedy", "probabilistic"],
        default="probabilistic",
        help="Strategy for selecting features (default: probabilistic)",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="/work3/monka/SummerSchool2025/results/EfficientNetB2_FocalLossLess_3FoldCV/",
        help="Directory containing trained models",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="/work3/monka/SummerSchool2025/active_shoppinglist.csv",
        help="Output file for shopping list",
    )
    parser.add_argument(
        "--diversity_factor",
        type=float,
        default=0.1,
        help="Factor to encourage class diversity in selection",
    )
    parser.add_argument(
        "--use_3fold_cv",
        action="store_true",
        default=True,
        help="Use 3-fold CV models for better loss estimation",
    )

    args = parser.parse_args()

    # Paths
    metadata_file = "/work3/monka/SummerSchool2025/metadata.csv"
    image_path = "/work3/monka/SummerSchool2025/FungiImages/"

    # Check if 3-fold CV models exist
    if args.use_3fold_cv:
        print("Using 3-fold CV models for better loss estimation")
        required_models = [f"best_accuracy_validation{i}.pth" for i in range(1, 4)]
        missing_models = []
        for model_file in required_models:
            model_path = os.path.join(args.results_dir, model_file)
            if not os.path.exists(model_path):
                missing_models.append(model_path)

        if missing_models:
            print("Error: 3-fold CV models not found:")
            for model in missing_models:
                print(f"  {model}")
            print(
                "Please train 3-fold CV models first using fungi_network_b2_3fold_cv_model_training.py"
            )
            return
    else:
        # Check if single model exists
        model_path = os.path.join(args.results_dir, "best_accuracy.pth")
        if not os.path.exists(model_path):
            print(f"Error: Model not found at {model_path}")
            print("Please train a model first")
            return

    # Initialize selector
    selector = ActiveFeatureSelector(
        metadata_file, image_path, args.results_dir, use_3fold_cv=args.use_3fold_cv
    )

    # Select samples and features
    acquisitions = selector.select_samples_and_features(
        budget=args.budget,
        strategy=args.strategy,
        diversity_factor=args.diversity_factor,
        feature_strategy=args.feature_strategy,
    )

    # Create shopping list
    selector.create_shopping_list(acquisitions, args.output_file)


if __name__ == "__main__":
    main()
