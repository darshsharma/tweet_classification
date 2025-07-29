#!/usr/bin/env python3
"""
Example usage of the DatasetPreparation class.

This script demonstrates how to use the data preparation pipeline
to create train/test splits while maintaining class balance.
"""

from data_preparation import DatasetPreparation

def simple_example():
    """
    Simple example showing basic usage of the DatasetPreparation class.
    """
    print("Simple Data Preparation Example")
    print("="*40)
    
    # Initialize the data preparation
    data_prep = DatasetPreparation('Q2_20230202_majority.csv')
    
    # Load and explore data
    data_prep.load_data()
    data_prep.explore_data()
    
    # Preprocess data
    data_prep.preprocess_data()
    
    # Create stratified split (80% train, 20% test)
    train_data, test_data = data_prep.create_stratified_split(test_size=0.2)
    
    # Analyze the split
    data_prep.analyze_split_distribution()
    
    # Save the datasets
    data_prep.save_datasets()
    
    print("\nData preparation completed!")
    print("Check the generated files:")
    print("- train_data.csv")
    print("- test_data.csv")
    print("- label_mapping.csv")

def custom_split_example():
    """
    Example showing custom split parameters.
    """
    print("\nCustom Split Example")
    print("="*40)
    
    # Initialize with custom parameters
    data_prep = DatasetPreparation('Q2_20230202_majority.csv')
    
    # Load and preprocess
    data_prep.load_data()
    data_prep.preprocess_data()
    
    # Create custom split (70% train, 30% test)
    train_data, test_data = data_prep.create_stratified_split(
        test_size=0.3,  # 30% for testing
        random_state=123  # Different random seed
    )
    
    # Save with custom filenames
    data_prep.save_datasets(
        train_file='custom_train_70_30.csv',
        test_file='custom_test_70_30.csv'
    )
    
    print("Custom split completed!")
    print("Files created:")
    print("- custom_train_70_30.csv")
    print("- custom_test_70_30.csv")

if __name__ == "__main__":
    # Run the simple example
    simple_example()
    
    # Run the custom example
    custom_split_example() 