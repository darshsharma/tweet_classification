#!/usr/bin/env python3
"""
Test script for Flan-T5 evaluation with a small sample.

This script tests the Flan-T5 evaluation on a small subset of the test data
to verify everything works correctly before running the full evaluation.
"""

from flan_t5_evaluation import FlanT5Evaluator
import pandas as pd

def test_single_prediction():
    """
    Test a single prediction to verify the model works.
    """
    print("Testing single prediction...")
    
    # Initialize evaluator
    evaluator = FlanT5Evaluator()
    
    # Test tweet
    test_tweet = "I got my COVID vaccine today and I feel great! Everyone should get vaccinated."
    
    # Get prediction
    predicted_label, raw_response = evaluator.predict_single_tweet(test_tweet)
    
    print(f"Test tweet: {test_tweet}")
    print(f"Predicted label: {predicted_label}")
    print(f"Raw response: {raw_response}")
    
    return predicted_label, raw_response

def test_small_sample():
    """
    Test on a small sample of the test data.
    """
    print("\nTesting on small sample...")
    
    # Initialize evaluator
    evaluator = FlanT5Evaluator()
    
    # Load test data
    test_data = evaluator.load_test_data()
    
    # Evaluate on small sample (10 samples)
    results = evaluator.evaluate_on_test_set(test_data, sample_size=10)
    
    # Print results
    evaluator.print_results(results)
    
    return results

def main():
    """
    Main function to run tests.
    """
    print("FLAN-T5 EVALUATION TEST")
    print("="*40)
    
    # Test single prediction
    test_single_prediction()
    
    # Test small sample
    results = test_small_sample()
    
    print("\n" + "="*40)
    print("TEST COMPLETED!")
    print("="*40)
    print("If the test runs successfully, you can run the full evaluation with:")
    print("python flan_t5_evaluation.py")

if __name__ == "__main__":
    main() 