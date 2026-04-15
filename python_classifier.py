# -*- coding: utf-8 -*-
"""

"""
MORB/OIB Mantle Reservoir Classifier

This script uses a trained H2O AutoML model to classify MORB (Mid-Ocean Ridge Basalt) and
OIB (Ocean Island Basalt) samples into mantle reservoir classes based on trace element geochemistry.

Requirements:
    - h2o
    - pandas

Input features (6 columns required):
    U/Pb  : U/Pb ratio (dimensionless)
    Nb    : Nb concentration (ppm)
    Th/U  : Th/U ratio (dimensionless)
    La    : La concentration (ppm)
    Rb    : Rb concentration (ppm)
    Ba    : Ba concentration (ppm)

The script will prompt you to choose between:
    1. Batch classification from a CSV file
    2. Single sample classification with manual input

For CSV input, ensure your file has 6 columns in the following order:
    U/Pb, Nb, Th/U, La, Rb, Ba

Notes:
    - This model was trained with H2O version 3.46.0.10. It is recommended
      to run this script with the same version. Pin it in requirements.txt.
    - The saved model is a binary H2O model. For cross-version portability, a MOJO export
      is preferred (model.download_mojo()). Contact the authors if you need a MOJO version.
    - Output class labels: DM, EM1, EM2, HIMU
"""

import os
import glob
import h2o
import pandas as pd


FEATURE_NAMES = ["U/Pb", "Nb", "Th/U", "La", "Rb", "Ba"]


def load_model():
    """
    Locate and load the H2O model from the 'saved_model' subdirectory.
    Searches for any file in that directory, so the model can retain its
    H2O-generated filename. If multiple files are present, the first is used.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, "saved_model")

    if not os.path.isdir(model_dir):
        raise FileNotFoundError(
            f"Model directory not found at '{model_dir}'. "
            "Please ensure the 'saved_model' directory exists and contains the model file."
        )

    model_files = glob.glob(os.path.join(model_dir, "*"))
    # Filter out hidden files and subdirectories
    model_files = [f for f in model_files if os.path.isfile(f) and not os.path.basename(f).startswith(".")]

    if not model_files:
        raise FileNotFoundError(
            f"No model file found in '{model_dir}'. "
            "Please place the H2O model file in the 'saved_model' directory."
        )

    if len(model_files) > 1:
        print(f"Warning: Multiple files found in 'saved_model/'. Loading: {os.path.basename(model_files[0])}")

    model_path = model_files[0]
    print(f"Loading model from: {model_path}")
    return h2o.load_model(model_path)


def detect_numeric_header(columns):
    """
    Returns True if all column names look like numeric values (i.e., the CSV
    has no header row). pandas always reads headers as strings, so we check
    whether each column name string is numeric.
    """
    return all(str(c).lstrip("-").replace(".", "", 1).isdigit() for c in columns)


def validate_and_align_columns(input_data):
    """
    Ensures input_data has exactly the 6 expected feature columns in the correct
    order. If the CSV had no header, assigns default column names. If column names
    are present but misordered, reorders them. Raises an error if required columns
    are missing.
    """
    # Case 1: No header detected — assign expected column names by position
    if detect_numeric_header(input_data.columns):
        if input_data.shape[1] != len(FEATURE_NAMES):
            raise ValueError(
                f"Expected {len(FEATURE_NAMES)} columns, but got {input_data.shape[1]}. "
                f"Please ensure your CSV contains the following columns in order:\n"
                f"{FEATURE_NAMES}"
            )
        input_data.columns = FEATURE_NAMES
        print(f"No header detected. Assigned column names: {FEATURE_NAMES}")
        return input_data

    # Case 2: Header present — check column count
    if input_data.shape[1] != len(FEATURE_NAMES):
        raise ValueError(
            f"Expected {len(FEATURE_NAMES)} columns, but got {input_data.shape[1]}. "
            f"Please ensure your CSV contains the following columns:\n{FEATURE_NAMES}"
        )

    # Case 3: Correct columns, wrong order — reorder
    if set(input_data.columns) == set(FEATURE_NAMES):
        if list(input_data.columns) != FEATURE_NAMES:
            input_data = input_data[FEATURE_NAMES]
            print("Columns reordered to match expected feature order.")
    else:
        missing = set(FEATURE_NAMES) - set(input_data.columns)
        extra = set(input_data.columns) - set(FEATURE_NAMES)
        raise ValueError(
            f"Column mismatch.\n"
            f"  Missing columns : {sorted(missing)}\n"
            f"  Unexpected columns: {sorted(extra)}\n"
            f"  Expected        : {FEATURE_NAMES}"
        )

    return input_data


def warn_if_nan(input_data):
    """Warn the user if any NaN values are present after numeric coercion."""
    nan_counts = input_data.isnull().sum()
    if nan_counts.any():
        print("\nWarning: NaN values detected in the following columns after coercion:")
        for col, count in nan_counts[nan_counts > 0].items():
            print(f"  {col}: {count} missing value(s)")
        print("Rows with missing data will still be passed to the model but predictions may be unreliable.\n")


def build_prediction_output(input_data, predictions_df):
    """
    Appends predicted class and per-class probability columns to input_data.
    Returns the augmented DataFrame.
    """
    result = input_data.copy()
    result["Predicted_Reservoir"] = predictions_df["predict"].values
    prob_cols = [c for c in predictions_df.columns if c != "predict"]
    for col in prob_cols:
        result[f"P({col})"] = predictions_df[col].values
    return result


def classify_samples_from_csv(csv_file_path):
    """
    Loads a CSV of geochemical samples, runs batch classification using the
    saved H2O model, and returns a DataFrame with predicted classes and
    per-class probabilities appended.
    """
    h2o_model = load_model()

    try:
        input_data = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find the CSV file at: '{csv_file_path}'")

    # Validate and align columns
    input_data = validate_and_align_columns(input_data)

    # Coerce all feature columns to numeric, replacing non-numeric entries with NaN
    for col in FEATURE_NAMES:
        input_data[col] = pd.to_numeric(input_data[col], errors="coerce")

    # Warn user about any missing values introduced by coercion
    warn_if_nan(input_data)

    # Convert to H2OFrame and enforce numeric types
    input_h2o = h2o.H2OFrame(input_data)
    for col in FEATURE_NAMES:
        input_h2o[col] = input_h2o[col].asnumeric()

    # Run predictions
    predictions = h2o_model.predict(input_h2o)
    predictions_df = predictions.as_data_frame()

    return build_prediction_output(input_data, predictions_df)


def classify_single_sample():
    """
    Prompts the user to enter values for each feature interactively,
    runs classification, and prints the predicted class with probabilities.
    """
    h2o_model = load_model()

    sample_data = {}
    print("\nEnter trace element data for your sample:")
    for feature in FEATURE_NAMES:
        while True:
            try:
                value = float(input(f"  {feature}: "))
                sample_data[feature] = value
                break
            except ValueError:
                print("  Invalid input. Please enter a numeric value.")

    input_df = pd.DataFrame([sample_data])
    input_h2o = h2o.H2OFrame(input_df)
    for col in FEATURE_NAMES:
        input_h2o[col] = input_h2o[col].asnumeric()

    predictions = h2o_model.predict(input_h2o)
    predictions_df = predictions.as_data_frame()

    predicted_class = predictions_df["predict"].iloc[0]

    print(f"\nPredicted mantle reservoir: {predicted_class}")

    prob_cols = [c for c in predictions_df.columns if c != "predict"]
    if prob_cols:
        print("Class probabilities:")
        for col in prob_cols:
            print(f"  P({col}) = {predictions_df[col].iloc[0]:.4f}")

    return predicted_class


def main():
    h2o.init()

    try:
        print("=" * 45)
        print("   Mantle Reservoir Classifier")
        print("=" * 45)
        print("1. Classify multiple samples from a CSV file")
        print("2. Classify a single sample via manual input")
        print("-" * 45)

        choice = input("Enter your choice (1 or 2): ").strip()

        if choice == "1":
            csv_path = input("Path to input CSV file: ").strip()
            results = classify_samples_from_csv(csv_path)

            print(f"\nClassification complete. {len(results)} sample(s) processed.")
            print(results[["Predicted_Reservoir"] + [c for c in results.columns if c.startswith("P(")]].to_string(index=False))

            output_path = input("\nPath to save results CSV (including filename): ").strip()
            results.to_csv(output_path, index=False)
            print(f"Results saved to: {output_path}")

        elif choice == "2":
            classify_single_sample()

        else:
            print("Invalid choice. Please run the program again and enter 1 or 2.")

    except FileNotFoundError as e:
        print(f"\nFile not found: {e}")
    except ValueError as e:
        print(f"\nData error: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        raise
    finally:
        h2o.cluster().shutdown()


if __name__ == "__main__":
    main()
