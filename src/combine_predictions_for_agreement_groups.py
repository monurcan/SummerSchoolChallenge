#!/usr/bin/env python3
"""
Script to combine predictions for agreement groups by averaging probabilities.

For each agreement group, we:
1. Average the probabilities for all samples in that group
2. Take argmax to get the final prediction
3. For samples not in any agreement group, keep original predictions unchanged

Input files:
- test_predictions.csv: Original predictions (session_name on first line, then filename,prediction pairs)
- test_probabilities.csv: Probabilities (header with session_name and class_0-182, then filename,probs)
- metadata_clip_agreement_groups.csv: Agreement groups with filenames separated by semicolons

Output:
- combined_test_predictions.csv: Combined predictions in the same format as test_predictions.csv
"""

import pandas as pd
import numpy as np
import csv
import os


def load_test_predictions(predictions_file):
    """Load test predictions from CSV file."""
    predictions = {}
    with open(predictions_file, "r") as f:
        reader = csv.reader(f)
        session_name = next(reader)[0]  # First line is session name

        for row in reader:
            filename, prediction = row[0], int(row[1])
            predictions[filename] = prediction

    return session_name, predictions


def load_test_probabilities(probabilities_file):
    """Load test probabilities from CSV file."""
    probabilities = {}
    with open(probabilities_file, "r") as f:
        reader = csv.reader(f)
        header = next(reader)  # First line is header
        session_name = header[0]

        for row in reader:
            filename = row[0]
            probs = [float(x) for x in row[1:]]  # Skip filename, convert to float
            probabilities[filename] = np.array(probs)

    return session_name, probabilities


def load_agreement_groups(groups_file):
    """Load agreement groups from CSV file."""
    df = pd.read_csv(groups_file)
    groups = []

    for _, row in df.iterrows():
        filenames = row["filenames"].split(";")
        groups.append(
            {
                "group_id": row["group_id"],
                "filenames": filenames,
                "sample_count": row["sample_count"],
            }
        )

    return groups


def combine_predictions_for_groups(predictions, probabilities, agreement_groups):
    """
    Combine predictions for agreement groups by averaging probabilities.

    Returns:
        combined_predictions: Dict with updated predictions
        files_in_groups: Set of filenames that were in agreement groups
        changed_predictions: Dict tracking which files had their predictions changed
    """
    combined_predictions = predictions.copy()
    files_in_groups = set()
    changed_predictions = {}

    print(f"Processing {len(agreement_groups)} agreement groups...")

    for group in agreement_groups:
        group_id = group["group_id"]
        filenames = group["filenames"]

        # Find which files from this group exist in our probabilities
        existing_files = [f for f in filenames if f in probabilities]

        if len(existing_files) == 0:
            print(f"Warning: Group {group_id} has no files in probabilities data")
            continue

        if len(existing_files) != len(filenames):
            missing_files = set(filenames) - set(existing_files)
            print(f"Warning: Group {group_id} missing files: {missing_files}")

        # Average probabilities for files in this group
        group_probs = []
        for filename in existing_files:
            if filename in probabilities:
                group_probs.append(probabilities[filename])

        if len(group_probs) > 0:
            # Average probabilities across all samples in the group
            avg_probs = np.mean(group_probs, axis=0)

            # Get the prediction (argmax)
            combined_prediction = np.argmax(avg_probs)

            # Count how many files in this group will change predictions
            group_changes = 0
            original_predictions = []

            # Update predictions for all files in this group
            for filename in existing_files:
                original_pred = combined_predictions[filename]
                original_predictions.append(original_pred)

                if original_pred != combined_prediction:
                    changed_predictions[filename] = {
                        "old": original_pred,
                        "new": combined_prediction,
                        "group_id": group_id,
                    }
                    group_changes += 1

                combined_predictions[filename] = combined_prediction
                files_in_groups.add(filename)

            print(
                f"Group {group_id}: {len(existing_files)} files -> prediction {combined_prediction} "
                f"(original predictions: {original_predictions}, {group_changes} changed)"
            )

    return combined_predictions, files_in_groups, changed_predictions


def save_combined_predictions(combined_predictions, session_name, output_file):
    """Save combined predictions to CSV file."""
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)

        # Write session name as first line
        writer.writerow([session_name + "_combined"])

        # Write filename,prediction pairs
        for filename, prediction in combined_predictions.items():
            writer.writerow([filename, prediction])

    print(f"Combined predictions saved to: {output_file}")


def main():
    # File paths - you can modify these as needed
    base_dir = "/work3/monka/SummerSchool2025"

    # Input files
    predictions_file = os.path.join(
        base_dir,
        "results/EfficientNet_V2L_CrossEntropy_New/test_predictions_tta_10.csv",
    )
    probabilities_file = os.path.join(
        base_dir, "results/EfficientNet_V2L_CrossEntropy_New/test_probabilities.csv"
    )
    groups_file = os.path.join(base_dir, "metadata_clip_agreement_groups.csv")

    # Output file
    output_file = os.path.join(
        base_dir,
        "results/EfficientNet_V2L_CrossEntropy_New/combined_tta_10_test_predictions.csv",
    )

    # Check if input files exist
    for file_path in [predictions_file, probabilities_file, groups_file]:
        if not os.path.exists(file_path):
            print(f"Error: File not found: {file_path}")
            return

    print("Loading data...")

    # Load data
    session_name, predictions = load_test_predictions(predictions_file)
    _, probabilities = load_test_probabilities(probabilities_file)
    agreement_groups = load_agreement_groups(groups_file)

    print(f"Loaded {len(predictions)} predictions")
    print(f"Loaded {len(probabilities)} probability vectors")
    print(f"Loaded {len(agreement_groups)} agreement groups")

    # Combine predictions for agreement groups
    combined_predictions, files_in_groups, changed_predictions = (
        combine_predictions_for_groups(predictions, probabilities, agreement_groups)
    )

    # Save results
    save_combined_predictions(combined_predictions, session_name, output_file)

    # Print statistics
    total_files = len(predictions)
    files_changed = len(changed_predictions)
    files_in_groups_count = len(files_in_groups)
    files_unchanged = total_files - files_changed

    print("\nSummary:")
    print(f"Total files: {total_files}")
    print(f"Files in agreement groups: {files_in_groups_count}")
    print(f"Files with changed predictions: {files_changed}")
    print(f"Files with unchanged predictions: {files_unchanged}")
    print(
        f"Change rate in agreement groups: {files_changed}/{files_in_groups_count} = {files_changed / files_in_groups_count * 100:.1f}%"
        if files_in_groups_count > 0
        else "No files in agreement groups"
    )

    # Show examples of changes
    if changed_predictions:
        print("\nExample prediction changes:")
        count = 0
        for filename, change_info in changed_predictions.items():
            if count >= 10:  # Show max 10 examples
                break
            print(
                f"  {filename}: {change_info['old']} -> {change_info['new']} (Group {change_info['group_id']})"
            )
            count += 1
    else:
        print("\nNo predictions were changed by the agreement group strategy.")


if __name__ == "__main__":
    main()
