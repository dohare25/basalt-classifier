# basalt-classifier

## Description
A machine-learning-based classifier (Stacked Ensemble) for predicting the mantle reservoir of basalt samples from trace element (TE) geochemistry.

## Installation
1. Install H2O and dependencies: pip install -r requirements.txt
2. Download the trained model file ('classifier') and place it in a `saved_model/` directory

## Usage
1. **Open Python and run the code from the following Python script:** python_classifier.py
2. **Follow the prompts:**
- Choose option 1 to classify multiple samples from a CSV file
- Choose option 2 to classify a single sample by entering values manually
3. **For CSV input (Option 1):**
- Prepare a CSV file with 6 columns containing the required trace element data in ppm (number values only) (include labels for each trace element variable in the first row)
- (Refer to sample_input.csv for an example.)
- When prompted, enter the full path to your CSV file
- Specify the directory to which you want to save the results
- New CSV file will be created with predicted reservoir and probabilities for each sample
4. **For single sample input (Option 2):**
- The program will prompt you to enter each trace element or ratio value
- The predicted mantle reservoir (DM, EM1, EM2, or HIMU) will be displayed on screen
