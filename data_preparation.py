import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class DatasetPreparation:
    """
    A class to prepare the tweet dataset for training and testing.
    Handles data loading, preprocessing, and stratified train/test splitting.
    """
    
    def __init__(self, csv_file_path):
        """
        Initialize the dataset preparation.
        
        Args:
            csv_file_path (str): Path to the CSV file containing the dataset
        """
        self.csv_file_path = csv_file_path
        self.data = None
        self.label_encoder = LabelEncoder()
        self.train_data = None
        self.test_data = None
        
    def load_data(self):
        """
        Load the dataset from CSV file.
        
        Returns:
            pd.DataFrame: Loaded dataset
        """
        print("Loading dataset...")
        self.data = pd.read_csv(self.csv_file_path)
        print(f"Dataset loaded successfully!")
        print(f"Total samples: {len(self.data)}")
        print(f"Columns: {list(self.data.columns)}")
        return self.data
    
    def explore_data(self):
        """
        Explore the dataset to understand its structure and class distribution.
        """
        print("\n" + "="*50)
        print("DATASET EXPLORATION")
        print("="*50)
        
        # Basic info
        print(f"Dataset shape: {self.data.shape}")
        print(f"Memory usage: {self.data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Check for missing values
        print("\nMissing values:")
        missing_values = self.data.isnull().sum()
        for col, missing in missing_values.items():
            if missing > 0:
                print(f"  {col}: {missing} ({missing/len(self.data)*100:.2f}%)")
        
        # Class distribution
        print("\nClass distribution:")
        class_counts = self.data['label_true'].value_counts()
        for label, count in class_counts.items():
            percentage = count / len(self.data) * 100
            print(f"  {label}: {count} samples ({percentage:.2f}%)")
        
        # Display sample tweets from each class
        print("\nSample tweets from each class:")
        for label in self.data['label_true'].unique():
            sample_tweet = self.data[self.data['label_true'] == label]['tweet'].iloc[0]
            print(f"\n  {label.upper()}:")
            print(f"    {sample_tweet[:100]}...")
    
    def preprocess_data(self):
        """
        Preprocess the data for training.
        
        Returns:
            pd.DataFrame: Preprocessed dataset
        """
        print("\n" + "="*50)
        print("DATA PREPROCESSING")
        print("="*50)
        
        # Create a copy to avoid modifying original data
        processed_data = self.data.copy()
        
        # Remove rows with missing labels
        initial_count = len(processed_data)
        processed_data = processed_data.dropna(subset=['label_true'])
        final_count = len(processed_data)
        
        if initial_count != final_count:
            print(f"Removed {initial_count - final_count} rows with missing labels")
        
        # Clean tweet text (basic cleaning)
        processed_data['tweet_clean'] = processed_data['tweet'].astype(str).str.strip()
        
        # Remove empty tweets
        processed_data = processed_data[processed_data['tweet_clean'].str.len() > 0]
        print(f"Final dataset size after preprocessing: {len(processed_data)}")
        
        # Encode labels
        processed_data['label_encoded'] = self.label_encoder.fit_transform(processed_data['label_true'])
        
        # Display label mapping
        print("\nLabel encoding:")
        for i, label in enumerate(self.label_encoder.classes_):
            print(f"  {label} -> {i}")
        
        self.data = processed_data
        return processed_data
    
    def create_stratified_split(self, test_size=0.2, random_state=42):
        """
        Create stratified train/test split to maintain class balance.
        
        Args:
            test_size (float): Proportion of dataset to include in test split
            random_state (int): Random state for reproducibility
            
        Returns:
            tuple: (train_data, test_data)
        """
        print("\n" + "="*50)
        print("CREATING STRATIFIED TRAIN/TEST SPLIT")
        print("="*50)
        
        # Prepare features and labels
        X = self.data[['tweet_id', 'tweet', 'tweet_clean', 'created_at', 'month']]
        y = self.data['label_encoded']
        
        # Perform stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=y
        )
        
        # Reconstruct train and test datasets
        self.train_data = X_train.copy()
        self.train_data['label_true'] = self.label_encoder.inverse_transform(y_train)
        self.train_data['label_encoded'] = y_train
        
        self.test_data = X_test.copy()
        self.test_data['label_true'] = self.label_encoder.inverse_transform(y_test)
        self.test_data['label_encoded'] = y_test
        
        print(f"Train set size: {len(self.train_data)} ({len(self.train_data)/len(self.data)*100:.1f}%)")
        print(f"Test set size: {len(self.test_data)} ({len(self.test_data)/len(self.data)*100:.1f}%)")
        
        return self.train_data, self.test_data
    
    def analyze_split_distribution(self):
        """
        Analyze the class distribution in train and test sets.
        """
        print("\n" + "="*50)
        print("SPLIT DISTRIBUTION ANALYSIS")
        print("="*50)
        
        # Train set distribution
        print("Train set class distribution:")
        train_counts = self.train_data['label_true'].value_counts()
        for label, count in train_counts.items():
            percentage = count / len(self.train_data) * 100
            print(f"  {label}: {count} samples ({percentage:.2f}%)")
        
        # Test set distribution
        print("\nTest set class distribution:")
        test_counts = self.test_data['label_true'].value_counts()
        for label, count in test_counts.items():
            percentage = count / len(self.test_data) * 100
            print(f"  {label}: {count} samples ({percentage:.2f}%)")
        
        # Verify stratification worked correctly
        print("\nStratification verification:")
        for label in self.data['label_true'].unique():
            train_ratio = len(self.train_data[self.train_data['label_true'] == label]) / len(self.train_data)
            test_ratio = len(self.test_data[self.test_data['label_true'] == label]) / len(self.test_data)
            original_ratio = len(self.data[self.data['label_true'] == label]) / len(self.data)
            
            print(f"  {label}:")
            print(f"    Original: {original_ratio:.3f}")
            print(f"    Train: {train_ratio:.3f}")
            print(f"    Test: {test_ratio:.3f}")
            print(f"    Difference: {abs(train_ratio - test_ratio):.3f}")
    
    def save_datasets(self, train_file='train_data.csv', test_file='test_data.csv'):
        """
        Save the train and test datasets to CSV files.
        
        Args:
            train_file (str): Filename for train dataset
            test_file (str): Filename for test dataset
        """
        print("\n" + "="*50)
        print("SAVING DATASETS")
        print("="*50)
        
        # Save train dataset
        self.train_data.to_csv(train_file, index=False)
        print(f"Train dataset saved to: {train_file}")
        
        # Save test dataset
        self.test_data.to_csv(test_file, index=False)
        print(f"Test dataset saved to: {test_file}")
        
        # Save label encoder mapping
        label_mapping = pd.DataFrame({
            'label': self.label_encoder.classes_,
            'encoded_value': range(len(self.label_encoder.classes_))
        })
        label_mapping.to_csv('label_mapping.csv', index=False)
        print("Label mapping saved to: label_mapping.csv")
    
    def create_visualizations(self):
        """
        Create visualizations for the dataset and split.
        """
        print("\n" + "="*50)
        print("CREATING VISUALIZATIONS")
        print("="*50)
        
        # Set up the plotting style
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Original class distribution
        original_counts = self.data['label_true'].value_counts()
        axes[0, 0].pie(original_counts.values, labels=original_counts.index, autopct='%1.1f%%')
        axes[0, 0].set_title('Original Dataset Class Distribution')
        
        # 2. Train set distribution
        train_counts = self.train_data['label_true'].value_counts()
        axes[0, 1].pie(train_counts.values, labels=train_counts.index, autopct='%1.1f%%')
        axes[0, 1].set_title('Train Set Class Distribution')
        
        # 3. Test set distribution
        test_counts = self.test_data['label_true'].value_counts()
        axes[1, 0].pie(test_counts.values, labels=test_counts.index, autopct='%1.1f%%')
        axes[1, 0].set_title('Test Set Class Distribution')
        
        # 4. Comparison bar chart
        comparison_data = pd.DataFrame({
            'Original': original_counts,
            'Train': train_counts,
            'Test': test_counts
        }).fillna(0)
        
        comparison_data.plot(kind='bar', ax=axes[1, 1])
        axes[1, 1].set_title('Class Distribution Comparison')
        axes[1, 1].set_xlabel('Classes')
        axes[1, 1].set_ylabel('Number of Samples')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('dataset_analysis.png', dpi=300, bbox_inches='tight')
        print("Visualization saved to: dataset_analysis.png")
        plt.show()
    
    def generate_summary_report(self):
        """
        Generate a comprehensive summary report of the data preparation process.
        """
        print("\n" + "="*50)
        print("SUMMARY REPORT")
        print("="*50)
        
        report = f"""
DATASET PREPARATION SUMMARY REPORT
==================================

Dataset Information:
- Original file: {self.csv_file_path}
- Total samples: {len(self.data)}
- Features: {len(self.data.columns)}
- Classes: {len(self.data['label_true'].unique())}

Class Distribution:
"""
        for label, count in self.data['label_true'].value_counts().items():
            percentage = count / len(self.data) * 100
            report += f"- {label}: {count} samples ({percentage:.2f}%)\n"
        
        report += f"""
Train/Test Split:
- Train samples: {len(self.train_data)} ({len(self.train_data)/len(self.data)*100:.1f}%)
- Test samples: {len(self.test_data)} ({len(self.test_data)/len(self.data)*100:.1f}%)
- Split ratio: 80% train / 20% test

Data Quality:
- Missing values handled: Yes
- Empty tweets removed: Yes
- Labels encoded: Yes
- Stratified split: Yes (maintains class balance)

Output Files:
- train_data.csv: Training dataset
- test_data.csv: Test dataset  
- label_mapping.csv: Label encoding mapping
- dataset_analysis.png: Visualization of distributions

Preprocessing Steps:
1. Loaded original CSV file
2. Removed rows with missing labels
3. Cleaned tweet text (removed leading/trailing whitespace)
4. Removed empty tweets
5. Encoded categorical labels
6. Performed stratified train/test split (80/20)
7. Verified class balance maintained in both splits
8. Saved processed datasets

The dataset is now ready for training and testing with proper class balance maintained.
"""
        
        print(report)
        
        # Save report to file
        with open('data_preparation_report.txt', 'w') as f:
            f.write(report)
        print("Detailed report saved to: data_preparation_report.txt")

def main():
    """
    Main function to run the complete data preparation pipeline.
    """
    # Initialize the dataset preparation
    data_prep = DatasetPreparation('Q2_20230202_majority.csv')
    
    # Run the complete pipeline
    data_prep.load_data()
    data_prep.explore_data()
    data_prep.preprocess_data()
    data_prep.create_stratified_split(test_size=0.2, random_state=42)
    data_prep.analyze_split_distribution()
    data_prep.save_datasets()
    data_prep.create_visualizations()
    data_prep.generate_summary_report()
    
    print("\n" + "="*50)
    print("DATA PREPARATION COMPLETED SUCCESSFULLY!")
    print("="*50)
    print("Your datasets are ready for training and testing.")
    print("Files created:")
    print("- train_data.csv: Training dataset")
    print("- test_data.csv: Test dataset")
    print("- label_mapping.csv: Label encoding")
    print("- dataset_analysis.png: Visualizations")
    print("- data_preparation_report.txt: Detailed report")

if __name__ == "__main__":
    main() 