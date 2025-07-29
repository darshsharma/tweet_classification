#!/usr/bin/env python3
"""
Demonstration script for Flan-T5 evaluation structure.

This script shows the expected structure and output format
without requiring the actual model download.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import time
from tqdm import tqdm

class FlanT5Demo:
    """
    A demonstration class showing the Flan-T5 evaluation structure.
    """
    
    def __init__(self):
        """
        Initialize the demo evaluator.
        """
        self.valid_labels = ['in-favor', 'against', 'neutral-or-unclear']
        self.classification_prompt = 'Here is the tweet. "{tweet}"  Please just give the final classification as "in-favor", "against", or "neutral-or-unclear".'
        
        print("Flan-T5-Large Demo Evaluator")
        print("Note: This is a demonstration without actual model loading")
        
    def load_test_data(self, test_file='test_data.csv'):
        """
        Load the prepared test dataset.
        """
        print(f"Loading test data from {test_file}...")
        test_data = pd.read_csv(test_file)
        print(f"Loaded {len(test_data)} test samples")
        
        # Display class distribution
        print("\nTest set class distribution:")
        class_counts = test_data['label_true'].value_counts()
        for label, count in class_counts.items():
            percentage = count / len(test_data) * 100
            print(f"  {label}: {count} samples ({percentage:.2f}%)")
        
        return test_data
    
    def simulate_prediction(self, tweet):
        """
        Simulate a prediction for demonstration purposes.
        """
        # Simple rule-based simulation for demo
        tweet_lower = tweet.lower()
        
        if any(word in tweet_lower for word in ['vaccine', 'vaccinated', 'get vaccinated', 'covid vaccine']):
            if any(word in tweet_lower for word in ['against', 'refuse', 'won\'t', 'don\'t want']):
                return 'against', 'against'
            elif any(word in tweet_lower for word in ['get', 'got', 'should', 'important', 'safe']):
                return 'in-favor', 'in-favor'
            else:
                return 'neutral-or-unclear', 'neutral-or-unclear'
        else:
            return 'neutral-or-unclear', 'neutral-or-unclear'
    
    def evaluate_on_test_set(self, test_data, sample_size=50):
        """
        Evaluate on test set with simulated predictions.
        """
        print(f"\nEvaluating on {sample_size} samples...")
        
        if sample_size:
            test_data = test_data.sample(n=sample_size, random_state=42)
        
        # Lists to store results
        true_labels = []
        predicted_labels = []
        raw_responses = []
        processing_times = []
        
        # Process each tweet
        for idx, row in tqdm(test_data.iterrows(), total=len(test_data), desc="Processing tweets"):
            tweet = row['tweet_clean']
            true_label = row['label_true']
            
            # Record start time
            start_time = time.time()
            
            # Get prediction (simulated)
            predicted_label, raw_response = self.simulate_prediction(tweet)
            
            # Record processing time
            processing_time = time.time() - start_time
            processing_times.append(processing_time)
            
            # Store results
            true_labels.append(true_label)
            predicted_labels.append(predicted_label)
            raw_responses.append(raw_response)
        
        # Calculate metrics
        results = self.calculate_metrics(true_labels, predicted_labels, processing_times)
        results['raw_responses'] = raw_responses
        
        return results
    
    def calculate_metrics(self, true_labels, predicted_labels, processing_times):
        """
        Calculate evaluation metrics.
        """
        # Calculate F1 score
        f1_macro = f1_score(true_labels, predicted_labels, average='macro')
        f1_weighted = f1_score(true_labels, predicted_labels, average='weighted')
        
        # Calculate per-class F1 scores
        f1_per_class = f1_score(true_labels, predicted_labels, average=None, labels=self.valid_labels)
        
        # Calculate accuracy
        accuracy = sum(1 for true, pred in zip(true_labels, predicted_labels) if true == pred) / len(true_labels)
        
        # Calculate average processing time
        avg_processing_time = np.mean(processing_times)
        
        results = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'f1_per_class': dict(zip(self.valid_labels, f1_per_class)),
            'avg_processing_time': avg_processing_time,
            'total_processing_time': sum(processing_times),
            'true_labels': true_labels,
            'predicted_labels': predicted_labels,
            'processing_times': processing_times
        }
        
        return results
    
    def print_results(self, results):
        """
        Print evaluation results.
        """
        print("\n" + "="*60)
        print("DEMO EVALUATION RESULTS")
        print("="*60)
        
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"F1 Score (Macro): {results['f1_macro']:.4f}")
        print(f"F1 Score (Weighted): {results['f1_weighted']:.4f}")
        print(f"Average Processing Time: {results['avg_processing_time']:.4f} seconds")
        print(f"Total Processing Time: {results['total_processing_time']:.2f} seconds")
        
        print("\nPer-class F1 Scores:")
        for label, f1 in results['f1_per_class'].items():
            print(f"  {label}: {f1:.4f}")
        
        # Print classification report
        print("\nDetailed Classification Report:")
        print(classification_report(
            results['true_labels'], 
            results['predicted_labels'],
            target_names=self.valid_labels
        ))
    
    def create_confusion_matrix(self, results, save_path='flan_t5_demo_confusion_matrix.png'):
        """
        Create and save confusion matrix visualization.
        """
        cm = confusion_matrix(results['true_labels'], results['predicted_labels'], labels=self.valid_labels)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.valid_labels,
                    yticklabels=self.valid_labels)
        plt.title('Flan-T5-Large Demo Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
        plt.show()
    
    def analyze_sample_predictions(self, results, test_data, num_samples=5):
        """
        Analyze sample predictions.
        """
        print(f"\n" + "="*60)
        print(f"SAMPLE PREDICTION ANALYSIS")
        print("="*60)
        
        # Get indices of correct and incorrect predictions
        correct_indices = [i for i, (true, pred) in enumerate(zip(results['true_labels'], results['predicted_labels'])) if true == pred]
        incorrect_indices = [i for i, (true, pred) in enumerate(zip(results['true_labels'], results['predicted_labels'])) if true != pred]
        
        print(f"Correct predictions: {len(correct_indices)}")
        print(f"Incorrect predictions: {len(incorrect_indices)}")
        
        # Show some correct predictions
        if correct_indices:
            print(f"\nSample Correct Predictions:")
            for i, idx in enumerate(correct_indices[:num_samples//2]):
                tweet = test_data.iloc[idx]['tweet_clean']
                true_label = results['true_labels'][idx]
                pred_label = results['predicted_labels'][idx]
                raw_response = results['raw_responses'][idx]
                
                print(f"\n{i+1}. ✓ Correct: {true_label}")
                print(f"   Tweet: {tweet[:100]}...")
                print(f"   Model response: {raw_response}")
        
        # Show some incorrect predictions
        if incorrect_indices:
            print(f"\nSample Incorrect Predictions:")
            for i, idx in enumerate(incorrect_indices[:num_samples//2]):
                tweet = test_data.iloc[idx]['tweet_clean']
                true_label = results['true_labels'][idx]
                pred_label = results['predicted_labels'][idx]
                raw_response = results['raw_responses'][idx]
                
                print(f"\n{i+1}. ✗ True: {true_label}, Predicted: {pred_label}")
                print(f"   Tweet: {tweet[:100]}...")
                print(f"   Model response: {raw_response}")

def main():
    """
    Main function to run the demo.
    """
    print("FLAN-T5-LARGE DEMO EVALUATION")
    print("="*60)
    
    # Initialize the demo evaluator
    evaluator = FlanT5Demo()
    
    # Load test data
    test_data = evaluator.load_test_data()
    
    # Evaluate on small sample
    results = evaluator.evaluate_on_test_set(test_data, sample_size=50)
    
    # Print results
    evaluator.print_results(results)
    
    # Create visualizations
    evaluator.create_confusion_matrix(results)
    
    # Analyze sample predictions
    evaluator.analyze_sample_predictions(results, test_data)
    
    print("\n" + "="*60)
    print("DEMO COMPLETED!")
    print("="*60)
    print("This was a demonstration with simulated predictions.")
    print("To run the actual Flan-T5 evaluation:")
    print("1. Install transformers: pip install transformers torch")
    print("2. Run: python flan_t5_evaluation.py")

if __name__ == "__main__":
    main() 