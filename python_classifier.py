# -*- coding: utf-8 -*-
"""
MORB/OIB Mantle Reservoir Classifier

This script uses a trained H2O AutoML model to classify MORB (Mid-Ocean Ridge Basalt) and 
OIB (Ocean Island Basalt) samples into mantle reservoir classes based on trace element geochemistry.

Requirements:
-H2O (h2o package)
-pandas
-numpy
-scikit-learn

The script will prompt you to choose between:
1. Batch classification from CSV file
2. Single sample classification with manual input

For CSV input, ensure your file has 14 columns in the following order:
Rb/Ba, Rb/Sr, Ba/Th, Th/U, Th/Rb, U/Pb, Rb/K, Ba/La, Rb/Nb, Ba, Nb, Rb, La, Ce
"""

import os
import h2o
import pandas as pd

# Initialize H2O cluster
h2o.init()

# Function to load the model
def load_model():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "saved_model", "classifier_general")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please ensure the model file is in the 'saved_model' directory.")
    
    return h2o.load_model(model_path)

def validate_column_order(input_data):
    expected_columns = ['Rb/Ba', 'Rb/Sr', 'Ba/Th', 'Th/U', 'Th/Rb',
                       'U/Pb', 'Rb/K', 'Ba/La', 'Rb/Nb', 'Ba', 'Nb', 'Rb', 'La', 'Ce']
    
    # If CSV has headers, check they match expected order
    if not input_data.columns.equals(pd.Index(expected_columns)):
        print("Warning: Column names don't match expected order.")
        print(f"Expected: {expected_columns}")
        print(f"Found: {list(input_data.columns)}")
        
        # Optionally, reorder columns if they exist but are in wrong order
        if set(input_data.columns) == set(expected_columns):
            input_data = input_data[expected_columns]
            print("Columns reordered to match expected format.")
    
    return input_data

# Function to classify OIB samples from CSV
def classify_samples_from_csv(csv_file_path):
    try:
        h2o_model = load_model()
        
        # Read the CSV file
        input_data = pd.read_csv(csv_file_path)
        
        # Check column count before processing
        if input_data.shape[1] != 14:
            raise ValueError(f"Expected 14 columns, but got {input_data.shape[1]}."
                           f"Please ensure your CSV file contains the following 14 columns in order:\n"
                           f"Rb/Ba, Rb/Sr, Ba/Th, Th/U, Th/Rb, U/Pb, Rb/K, Ba/La, Rb/Nb, Ba, Nb, Rb, La, Ce")
        
        # Validate column order if headers are present
        input_data = validate_column_order(input_data)
        
        # Ensure all columns are numeric with proper NAs
        for col in input_data.columns:
            input_data[col] = pd.to_numeric(input_data[col], errors='coerce')
        
        # Convert to H2O Frame with explicit column types
        input_h2o = h2o.H2OFrame(input_data)
        
        # Explicitly set all columns to numeric
        for col in input_h2o.columns:
            input_h2o[col] = input_h2o[col].asnumeric()
        
        # Make predictions
        predictions = h2o_model.predict(input_h2o)
        
        # Extract predicted class
        predicted_classes = predictions['predict'].as_data_frame().iloc[:, 0]
        
        # Add predictions to original data
        input_data['Predicted_Reservoir'] = predicted_classes
        
        return input_data
        
    except FileNotFoundError:
        print(f"Error: Could not find the CSV file at {csv_file_path}")
        raise
    except Exception as e:
        print(f"Error processing CSV file: {e}")
        raise

# Function to classify a single OIB sample
def classify_single_sample():
    h2o_model = load_model()
    
    feature_names = ['Rb/Ba', 'Rb/Sr', 'Ba/Th', 'Th/U', 'Th/Rb',
                     'U/Pb', 'Rb/K', 'Ba/La', 'Rb/Nb', 'Ba', 'Nb', 'Rb', 'La', 'Ce']
    
    sample_data = {}
    print("Please enter the trace element data for your OIB sample:")
    for feature in feature_names:
        while True:
            try:
                value = float(input(f"{feature}: "))
                sample_data[feature] = value
                break
            except ValueError:
                print("Invalid input. Please enter a numeric value.")

    input_df = pd.DataFrame([sample_data])
    
    # Convert input data to H2O Frame
    input_h2o = h2o.H2OFrame(input_df)
    
    prediction = h2o_model.predict(input_h2o)
    
    # Extract predicted class
    predicted_class = prediction['predict'].as_data_frame().iloc[0, 0]
    
    return predicted_class

# Main function
def main():
    try:
        print("OIB Mantle Reservoir Classifier")
        print("1. Classify multiple samples from a CSV file")
        print("2. Classify a single sample through manual input")
        
        choice = input("Enter your choice (1 or 2): ")
        
        if choice == '1':
            csv_path = input("Enter the path to your CSV file: ")
            results = classify_samples_from_csv(csv_path)
            output_path = input("Enter the path to save the results CSV (including file name): ")
            results.to_csv(output_path, index=False)
            print(f"Results saved to {output_path}")
        elif choice == '2':
            result = classify_single_sample()
            print(f"The predicted mantle reservoir is: {result}")
        else:
            print("Invalid choice. Please run the program again and select 1 or 2.")
    
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        h2o.cluster().shutdown()

if __name__ == "__main__":
    main()

