# Dataset Preparation for Tweet Classification

This repository contains code for preparing a tweet dataset for training and testing machine learning models. The code ensures proper class balance is maintained during the train/test split.

## Features

- **Stratified Train/Test Split**: Maintains the same class distribution in both train and test sets
- **Data Preprocessing**: Handles missing values, cleans text, and encodes labels
- **Comprehensive Analysis**: Provides detailed exploration and visualization of the dataset
- **Flexible Configuration**: Supports custom split ratios and random seeds
- **Complete Pipeline**: From data loading to saving processed datasets

## Dataset Information

The dataset contains tweets with the following structure:
- `tweet_id`: Unique identifier for each tweet
- `created_at`: Timestamp of the tweet
- `tweet`: The actual tweet text
- `label_true`: The true label (in-favor, against, neutral-or-unclear)
- `month`: Month information
- `label_pred`: Predicted label (empty in original data)

### Class Distribution
- `in-favor`: 1,371 samples (47.6%)
- `against`: 966 samples (33.5%)
- `neutral-or-unclear`: 565 samples (19.6%)

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the complete data preparation pipeline:

```bash
python data_preparation.py
```

This will:
1. Load the dataset from `Q2_20230202_majority.csv`
2. Explore and analyze the data
3. Preprocess the data (handle missing values, clean text)
4. Create a stratified 80/20 train/test split
5. Save the processed datasets
6. Generate visualizations and reports

### Custom Usage

You can also use the `DatasetPreparation` class directly:

```python
from data_preparation import DatasetPreparation

# Initialize
data_prep = DatasetPreparation('Q2_20230202_majority.csv')

# Load and explore
data_prep.load_data()
data_prep.explore_data()

# Preprocess
data_prep.preprocess_data()

# Create custom split (70% train, 30% test)
train_data, test_data = data_prep.create_stratified_split(
    test_size=0.3,
    random_state=123
)

# Save datasets
data_prep.save_datasets(
    train_file='my_train.csv',
    test_file='my_test.csv'
)
```

### Example Usage

Run the example script to see different usage patterns:

```bash
python example_usage.py
```

## Output Files

The script generates the following files:

1. **`train_data.csv`**: Training dataset (80% of data)
2. **`test_data.csv`**: Test dataset (20% of data)
3. **`label_mapping.csv`**: Mapping between original labels and encoded values
4. **`dataset_analysis.png`**: Visualizations showing class distributions
5. **`data_preparation_report.txt`**: Detailed report of the preparation process

## Key Features

### Stratified Splitting
The code uses scikit-learn's `train_test_split` with `stratify=y` to ensure that the class distribution in the train and test sets matches the original dataset.

### Data Quality Checks
- Removes rows with missing labels
- Cleans tweet text (removes leading/trailing whitespace)
- Removes empty tweets
- Encodes categorical labels for machine learning

### Visualization
Creates comprehensive visualizations including:
- Pie charts showing class distributions
- Bar charts comparing original vs split distributions
- Detailed analysis of stratification effectiveness

### Reproducibility
- Uses fixed random seeds for reproducible results
- Saves all intermediate data and mappings
- Generates detailed reports for documentation

## Class Structure

The `DatasetPreparation` class provides the following methods:

- `load_data()`: Load the CSV dataset
- `explore_data()`: Analyze dataset structure and class distribution
- `preprocess_data()`: Clean and prepare data for training
- `create_stratified_split()`: Create train/test split with class balance
- `analyze_split_distribution()`: Verify class balance in splits
- `save_datasets()`: Save processed datasets to files
- `create_visualizations()`: Generate plots and charts
- `generate_summary_report()`: Create detailed report

## Requirements

- Python 3.7+
- pandas >= 1.3.0
- numpy >= 1.21.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0

## Example Output

When you run the script, you'll see output like this:

```
Loading dataset...
Dataset loaded successfully!
Total samples: 2902
Columns: ['tweet_id', 'created_at', 'tweet', 'label_true', 'month', 'label_pred']

==================================================
DATASET EXPLORATION
==================================================
Dataset shape: (2902, 6)
Memory usage: 2.34 MB

Class distribution:
  in-favor: 1371 samples (47.24%)
  against: 966 samples (33.29%)
  neutral-or-unclear: 565 samples (19.47%)

==================================================
CREATING STRATIFIED TRAIN/TEST SPLIT
==================================================
Train set size: 2321 (80.0%)
Test set size: 581 (20.0%)

==================================================
SPLIT DISTRIBUTION ANALYSIS
==================================================
Train set class distribution:
  in-favor: 1097 samples (47.26%)
  against: 773 samples (33.31%)
  neutral-or-unclear: 451 samples (19.43%)

Test set class distribution:
  in-favor: 274 samples (47.16%)
  against: 193 samples (33.22%)
  neutral-or-unclear: 114 samples (19.62%)
```

## Notes

- The stratified split ensures that the proportion of samples for each class is preserved in both train and test sets
- The random seed ensures reproducible results across runs
- All preprocessing steps are documented and can be easily modified
- The code handles edge cases like missing values and empty tweets

## License

This code is provided as-is for educational and research purposes. 