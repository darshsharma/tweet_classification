#!/usr/bin/env python3
"""
Evaluation script for Flan-T5-large model on tweet classification task.

This script loads the prepared test dataset and evaluates the Flan-T5-large model
without any fine-tuning, using zero-shot classification.
"""

import pandas as pd
import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings('ignore')

class FlanT5Evaluator:
    """
    A class to evaluate Flan-T5-large model on tweet classification task.
    """
    
    def __init__(self, model_name="google/flan-t5-large", device=None):
        """
        Initialize the Flan-T5 evaluator.
        
        Args:
            model_name (str): Name of the Flan-T5 model to use
            device (str): Device to run the model on ('cuda', 'cpu', or None for auto)
        """
        self.model_name = model_name
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading {model_name}...")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded successfully on {self.device}")
        
        # Define the prompt template
        self.system_prompt = 'You are a helpful assistant that can classify tweets with respect to COVID-19 vaccine into three categories: "in-favor", "against", or "neutral-or-unclear".'
        self.classification_prompt = 'Here is the tweet. "{tweet}"  Please just give the final classification as "in-favor", "against", or "neutral-or-unclear".'
        
        # Define valid labels
        self.valid_labels = ['in-favor', 'against', 'neutral-or-unclear']
        
    def load_test_data(self, test_file='test_data.csv'):
        """
        Load the prepared test dataset.
        
        Args:
            test_file (str): Path to the test dataset CSV file
            
        Returns:
            pd.DataFrame: Test dataset
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
    
    def create_prompt(self, tweet):
        """
        Create the classification prompt for a given tweet.
        
        Args:
            tweet (str): The tweet text
            
        Returns:
            str: The complete prompt
        """
        return self.classification_prompt.format(tweet=tweet)
    
    def predict_single_tweet(self, tweet, max_length=50):
        """
        Predict the classification for a single tweet.
        
        Args:
            tweet (str): The tweet text
            max_length (int): Maximum length for generation
            
        Returns:
            str: Predicted label
        """
        # Create the prompt
        prompt = self.create_prompt(tweet)
        
        # Tokenize input
        input_ids = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).input_ids
        input_ids = input_ids.to(self.device)
        
        # Generate output
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_length=max_length,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode the output
        predicted_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        
        # Extract the predicted label
        predicted_label = self.extract_label_from_response(predicted_text)
        
        return predicted_label, predicted_text
    
    def extract_label_from_response(self, response):
        """
        Extract the predicted label from the model's response.
        
        Args:
            response (str): The model's response text
            
        Returns:
            str: Extracted label
        """
        # Convert to lowercase for matching
        response_lower = response.lower().strip()
        
        # Try to find exact matches
        for label in self.valid_labels:
            if label.lower() in response_lower:
                return label
        
        # If no exact match, try partial matches
        if 'favor' in response_lower or 'in-favor' in response_lower:
            return 'in-favor'
        elif 'against' in response_lower:
            return 'against'
        elif 'neutral' in response_lower or 'unclear' in response_lower:
            return 'neutral-or-unclear'
        
        # Default to neutral if no match found
        return 'neutral-or-unclear'
    
    def evaluate_on_test_set(self, test_data, sample_size=None):
        """
        Evaluate the model on the entire test set.
        
        Args:
            test_data (pd.DataFrame): Test dataset
            sample_size (int): Number of samples to evaluate (None for all)
            
        Returns:
            dict: Evaluation results
        """
        print(f"\nEvaluating model on test set...")
        
        if sample_size:
            test_data = test_data.sample(n=sample_size, random_state=42)
            print(f"Using {sample_size} samples for evaluation")
        
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
            
            # Get prediction
            predicted_label, raw_response = self.predict_single_tweet(tweet)
            
            # Record processing time
            processing_time = time.time() - start_time
            processing_times.append(processing_time)
            
            # Store results
            true_labels.append(true_label)
            predicted_labels.append(predicted_label)
            raw_responses.append(raw_response)
        
        # Calculate metrics
        results = self.calculate_metrics(true_labels, predicted_labels, processing_times)
        
        # Add raw responses to results
        results['raw_responses'] = raw_responses
        
        return results
    
    def calculate_metrics(self, true_labels, predicted_labels, processing_times):
        """
        Calculate evaluation metrics.
        
        Args:
            true_labels (list): True labels
            predicted_labels (list): Predicted labels
            processing_times (list): Processing times for each prediction
            
        Returns:
            dict: Evaluation metrics
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
        
        Args:
            results (dict): Evaluation results
        """
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"F1 Score (Macro): {results['f1_macro']:.4f}")
        print(f"F1 Score (Weighted): {results['f1_weighted']:.4f}")
        print(f"Average Processing Time: {results['avg_processing_time']:.2f} seconds")
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
    
    def create_confusion_matrix(self, results, save_path='flan_t5_confusion_matrix.png'):
        """
        Create and save confusion matrix visualization.
        
        Args:
            results (dict): Evaluation results
            save_path (str): Path to save the confusion matrix
        """
        cm = confusion_matrix(results['true_labels'], results['predicted_labels'], labels=self.valid_labels)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.valid_labels,
                    yticklabels=self.valid_labels)
        plt.title('Flan-T5-Large Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
        plt.show()
    
    def analyze_sample_predictions(self, results, test_data, num_samples=10):
        """
        Analyze sample predictions to understand model behavior.
        
        Args:
            results (dict): Evaluation results
            test_data (pd.DataFrame): Test dataset
            num_samples (int): Number of samples to analyze
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
    
    def save_results(self, results, save_path='flan_t5_evaluation_results.txt'):
        """
        Save evaluation results to a text file.
        
        Args:
            results (dict): Evaluation results
            save_path (str): Path to save the results
        """
        with open(save_path, 'w') as f:
            f.write("FLAN-T5-LARGE EVALUATION RESULTS\n")
            f.write("="*50 + "\n\n")
            
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Device: {self.device}\n")
            f.write(f"Total samples: {len(results['true_labels'])}\n\n")
            
            f.write(f"Accuracy: {results['accuracy']:.4f}\n")
            f.write(f"F1 Score (Macro): {results['f1_macro']:.4f}\n")
            f.write(f"F1 Score (Weighted): {results['f1_weighted']:.4f}\n")
            f.write(f"Average Processing Time: {results['avg_processing_time']:.2f} seconds\n")
            f.write(f"Total Processing Time: {results['total_processing_time']:.2f} seconds\n\n")
            
            f.write("Per-class F1 Scores:\n")
            for label, f1 in results['f1_per_class'].items():
                f.write(f"  {label}: {f1:.4f}\n")
            
            f.write("\nDetailed Classification Report:\n")
            f.write(classification_report(
                results['true_labels'], 
                results['predicted_labels'],
                target_names=self.valid_labels
            ))
        
        print(f"Results saved to: {save_path}")

def main():
    """
    Main function to run the Flan-T5 evaluation.
    """
    print("FLAN-T5-LARGE TWEET CLASSIFICATION EVALUATION")
    print("="*60)
    
    # Initialize the evaluator
    evaluator = FlanT5Evaluator()
    
    # Load test data
    test_data = evaluator.load_test_data()
    
    # Evaluate on test set (you can set sample_size to test on a subset first)
    # For full evaluation, set sample_size=None
    results = evaluator.evaluate_on_test_set(test_data, sample_size=None)
    
    # Print results
    evaluator.print_results(results)
    
    # Create visualizations
    evaluator.create_confusion_matrix(results)
    
    # Analyze sample predictions
    evaluator.analyze_sample_predictions(results, test_data)
    
    # Save results
    evaluator.save_results(results)
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETED!")
    print("="*60)
    print("Key Results:")
    print(f"- F1 Score (Macro): {results['f1_macro']:.4f}")
    print(f"- F1 Score (Weighted): {results['f1_weighted']:.4f}")
    print(f"- Accuracy: {results['accuracy']:.4f}")
    print("\nFiles created:")
    print("- flan_t5_confusion_matrix.png")
    print("- flan_t5_evaluation_results.txt")

if __name__ == "__main__":
    main() 