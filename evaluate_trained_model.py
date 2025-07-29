#!/usr/bin/env python3
"""
Evaluation script for the trained Flan-T5 model.

This script loads the trained model and evaluates it on the test dataset
to compare performance with the zero-shot baseline.
"""

import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings('ignore')
import os

class TrainedModelEvaluator:
    """
    A class to evaluate the trained Flan-T5 model on tweet classification task.
    """
    
    def __init__(self, model_path="flan_t5_stance_classifier", device=None):
        """
        Initialize the trained model evaluator.
        
        Args:
            model_path (str): Path to the trained model
            device (str): Device to run the model on ('cuda', 'cpu', or None for auto)
        """
        self.model_path = model_path
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading trained model from {model_path}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load base model
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            "google/flan-t5-large",
            torch_dtype=torch.float16,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # Load trained model
        self.model = PeftModel.from_pretrained(base_model, model_path)
        self.model.eval()
        
        print(f"Trained model loaded successfully on {self.device}")
        
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
        return (
            "Classify the stance of the following tweet as "
            "in-favor, against, or neutral-or-unclear.\n\nTweet:\n"
            f"{tweet}\n\nAnswer:"
        )
    
    def predict_single_tweet(self, tweet, max_length=16):
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
        input_ids = self.tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True).input_ids
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
        print(f"\nEvaluating trained model on test set...")
        
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
        print("TRAINED MODEL EVALUATION RESULTS")
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
    
    def create_confusion_matrix(self, results, save_path='trained_model_confusion_matrix.png'):
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
        plt.title('Trained Flan-T5-Large Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
        plt.show()
    
    def compare_with_baseline(self, baseline_results_file='flan_t5_evaluation_results.txt'):
        """
        Compare trained model results with zero-shot baseline.
        
        Args:
            baseline_results_file (str): Path to baseline results file
        """
        print("\n" + "="*60)
        print("COMPARISON WITH ZERO-SHOT BASELINE")
        print("="*60)
        
        # Load baseline results
        if os.path.exists(baseline_results_file):
            with open(baseline_results_file, 'r') as f:
                baseline_content = f.read()
            
            # Extract baseline metrics (simple parsing)
            baseline_accuracy = None
            baseline_f1_macro = None
            
            for line in baseline_content.split('\n'):
                if 'Accuracy:' in line:
                    baseline_accuracy = float(line.split(':')[1].strip())
                elif 'F1 Score (Macro):' in line:
                    baseline_f1_macro = float(line.split(':')[1].strip())
            
            if baseline_accuracy is not None and baseline_f1_macro is not None:
                print(f"Zero-shot Baseline:")
                print(f"  Accuracy: {baseline_accuracy:.4f}")
                print(f"  F1 Score (Macro): {baseline_f1_macro:.4f}")
                
                print(f"\nTrained Model:")
                print(f"  Accuracy: {self.current_results['accuracy']:.4f}")
                print(f"  F1 Score (Macro): {self.current_results['f1_macro']:.4f}")
                
                print(f"\nImprovement:")
                print(f"  Accuracy: {self.current_results['accuracy'] - baseline_accuracy:+.4f}")
                print(f"  F1 Score (Macro): {self.current_results['f1_macro'] - baseline_f1_macro:+.4f}")
            else:
                print("Could not parse baseline results file.")
        else:
            print("Baseline results file not found.")
    
    def save_results(self, results, save_path='trained_model_evaluation_results.txt'):
        """
        Save evaluation results to a text file.
        
        Args:
            results (dict): Evaluation results
            save_path (str): Path to save the results
        """
        with open(save_path, 'w') as f:
            f.write("TRAINED FLAN-T5-LARGE EVALUATION RESULTS\n")
            f.write("="*50 + "\n\n")
            
            f.write(f"Model: {self.model_path}\n")
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
    Main function to run the trained model evaluation.
    """
    print("TRAINED FLAN-T5-LARGE EVALUATION")
    print("="*60)
    
    # Check if trained model exists
    if not os.path.exists('flan_t5_stance_classifier'):
        print("Error: Trained model not found!")
        print("Please run the training script first:")
        print("python train_flan_t5.py")
        return
    
    # Initialize the evaluator
    evaluator = TrainedModelEvaluator()
    
    # Load test data
    test_data = evaluator.load_test_data()
    
    # Evaluate on test set
    results = evaluator.evaluate_on_test_set(test_data, sample_size=None)
    
    # Store results for comparison
    evaluator.current_results = results
    
    # Print results
    evaluator.print_results(results)
    
    # Create visualizations
    evaluator.create_confusion_matrix(results)
    
    # Compare with baseline
    evaluator.compare_with_baseline()
    
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
    print("- trained_model_confusion_matrix.png")
    print("- trained_model_evaluation_results.txt")

if __name__ == "__main__":
    main() 