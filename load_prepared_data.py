#!/usr/bin/env python3
"""
Utility script to load and use the prepared train/test datasets.

This script demonstrates how to load the processed datasets
and use them for machine learning tasks.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_prepared_datasets():
    """
    Load the prepared train and test datasets.
    
    Returns:
        tuple: (train_data, test_data, label_mapping)
    """
    print("Loading prepared datasets...")
    
    # Load train and test datasets
    train_data = pd.read_csv('train_data.csv')
    test_data = pd.read_csv('test_data.csv')
    label_mapping = pd.read_csv('label_mapping.csv')
    
    print(f"Train dataset: {len(train_data)} samples")
    print(f"Test dataset: {len(test_data)} samples")
    print(f"Label mapping: {len(label_mapping)} classes")
    
    return train_data, test_data, label_mapping

def verify_class_balance(train_data, test_data):
    """
    Verify that class balance is maintained in both datasets.
    """
    print("\n" + "="*50)
    print("CLASS BALANCE VERIFICATION")
    print("="*50)
    
    # Train set distribution
    print("Train set class distribution:")
    train_counts = train_data['label_true'].value_counts()
    for label, count in train_counts.items():
        percentage = count / len(train_data) * 100
        print(f"  {label}: {count} samples ({percentage:.2f}%)")
    
    # Test set distribution
    print("\nTest set class distribution:")
    test_counts = test_data['label_true'].value_counts()
    for label, count in test_counts.items():
        percentage = count / len(test_data) * 100
        print(f"  {label}: {count} samples ({percentage:.2f}%)")
    
    # Compare distributions
    print("\nDistribution comparison:")
    for label in train_counts.index:
        train_ratio = train_counts[label] / len(train_data)
        test_ratio = test_counts[label] / len(test_data)
        diff = abs(train_ratio - test_ratio)
        print(f"  {label}: Train={train_ratio:.3f}, Test={test_ratio:.3f}, Diff={diff:.3f}")

def demonstrate_ml_pipeline(train_data, test_data):
    """
    Demonstrate a simple machine learning pipeline using the prepared data.
    """
    print("\n" + "="*50)
    print("MACHINE LEARNING PIPELINE DEMONSTRATION")
    print("="*50)
    
    # Prepare features and labels
    X_train = train_data['tweet_clean']
    y_train = train_data['label_encoded']
    X_test = test_data['tweet_clean']
    y_test = test_data['label_encoded']
    
    print(f"Training on {len(X_train)} samples")
    print(f"Testing on {len(X_test)} samples")
    
    # Create TF-IDF features
    print("\nCreating TF-IDF features...")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"Feature matrix shape: {X_train_tfidf.shape}")
    
    # Train a simple classifier
    print("\nTraining Logistic Regression classifier...")
    classifier = LogisticRegression(random_state=42, max_iter=1000)
    classifier.fit(X_train_tfidf, y_train)
    
    # Make predictions
    y_pred = classifier.predict(X_test_tfidf)
    
    # Evaluate
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                              target_names=['against', 'in-favor', 'neutral-or-unclear']))
    
    # Create confusion matrix visualization
    print("\nCreating confusion matrix...")
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['against', 'in-favor', 'neutral-or-unclear'],
                yticklabels=['against', 'in-favor', 'neutral-or-unclear'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("Confusion matrix saved to: confusion_matrix.png")
    plt.show()

def analyze_sample_predictions(train_data, test_data, vectorizer, classifier):
    """
    Analyze some sample predictions to understand model behavior.
    """
    print("\n" + "="*50)
    print("SAMPLE PREDICTION ANALYSIS")
    print("="*50)
    
    # Get some sample predictions
    X_test = test_data['tweet_clean'].head(10)
    y_test = test_data['label_encoded'].head(10)
    
    X_test_tfidf = vectorizer.transform(X_test)
    y_pred = classifier.predict(X_test_tfidf)
    
    # Create label mapping
    label_map = {0: 'against', 1: 'in-favor', 2: 'neutral-or-unclear'}
    
    print("Sample predictions:")
    for i, (tweet, true_label, pred_label) in enumerate(zip(X_test, y_test, y_pred)):
        true_label_name = label_map[true_label]
        pred_label_name = label_map[pred_label]
        correct = "✓" if true_label == pred_label else "✗"
        
        print(f"\n{i+1}. {correct} True: {true_label_name}, Predicted: {pred_label_name}")
        print(f"   Tweet: {tweet[:100]}...")

def main():
    """
    Main function to demonstrate loading and using prepared datasets.
    """
    print("DATASET LOADING AND USAGE DEMONSTRATION")
    print("="*50)
    
    # Load datasets
    train_data, test_data, label_mapping = load_prepared_datasets()
    
    # Verify class balance
    verify_class_balance(train_data, test_data)
    
    # Demonstrate ML pipeline
    demonstrate_ml_pipeline(train_data, test_data)
    
    print("\n" + "="*50)
    print("DEMONSTRATION COMPLETED!")
    print("="*50)
    print("The prepared datasets are ready for your machine learning tasks.")
    print("Key files:")
    print("- train_data.csv: Training dataset")
    print("- test_data.csv: Test dataset")
    print("- label_mapping.csv: Label encoding")
    print("- confusion_matrix.png: Model evaluation")

if __name__ == "__main__":
    main() 