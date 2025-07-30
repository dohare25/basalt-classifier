# basalt-classifiers

## Description
Two machine-learning classifiers for predicting the mantle reservoir of basalt samples from trace element (TE) geochemistry.

## Installation
1. Install H2O and dependencies: pip install -r requirements.txt
2. Download the trained model file (`classifier_general` or 'classifier_HIMU') and place it in a `saved_model/` directory

## Usage
1. **Open Python and run the code from the following Python script:** python_classifier.py
2. **Follow the prompts:**
- Choose option 1 to classify multiple samples from a CSV file
- Choose option 2 to classify a single sample by entering values manually
3. **For CSV input (Option 1):**
- Prepare a CSV file with 14 columns containing the required trace element data (include labels for each trace element variable in the first row)
- When prompted, enter the full path to your CSV file
- Specify the directory to which you want to save the results
4. **For single sample input (Option 2):**
- The program will prompt you to enter each trace element or ratio value
- The predicted mantle reservoir (DM, EM1, EM2, or HIMU) will be displayed on screen
5. **To use the classifier optimized for HIMU**
-Modify the load_model function in the Python script: When defining the variable model_path, change "classifier_general" to "classifier_HIMU".
-This will load the second classifier, which uses different training data and is optimized to improve classification of the HIMU reservoir.
-Run the modified Python script on your trace element data to obtain mantle reservoir predictions
