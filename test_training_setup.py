#!/usr/bin/env python3
"""
Test script to verify the training setup works correctly.

This script tests the key components of the training pipeline without
actually running the full training process.
"""

import pandas as pd
import numpy as np
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import LoraConfig, get_peft_model, TaskType
import torch
import os
import warnings
warnings.filterwarnings('ignore')

def test_data_loading():
    """Test that the training data can be loaded correctly."""
    print("Testing data loading...")
    
    if not os.path.exists('train_data.csv'):
        print("❌ train_data.csv not found!")
        return False
    
    # Load the data
    train_data = pd.read_csv('train_data.csv')
    print(f"✅ Loaded {len(train_data)} training samples")
    
    # Check required columns
    required_columns = ['tweet_clean', 'label_true']
    missing_columns = [col for col in required_columns if col not in train_data.columns]
    
    if missing_columns:
        print(f"❌ Missing columns: {missing_columns}")
        return False
    
    print("✅ All required columns present")
    
    # Check class distribution
    class_counts = train_data['label_true'].value_counts()
    print("Class distribution:")
    for label, count in class_counts.items():
        print(f"  {label}: {count} samples")
    
    return True

def test_model_loading():
    """Test that the model and tokenizer can be loaded."""
    print("\nTesting model loading...")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
        print("✅ Tokenizer loaded successfully")
        
        # Add padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print("✅ Added padding token")
        
        # Load base model
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            "google/flan-t5-large",
            torch_dtype=torch.float16,
            device_map="auto" if torch.cuda.is_available() else None
        )
        print("✅ Base model loaded successfully")
        
        # Setup LoRA configuration
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q", "v"]
        )
        
        # Create PEFT model
        model = get_peft_model(base_model, peft_config)
        print("✅ LoRA model created successfully")
        
        # Print trainable parameters
        model.print_trainable_parameters()
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def test_data_processing():
    """Test the data processing pipeline."""
    print("\nTesting data processing...")
    
    try:
        # Load a small sample of data
        train_data = pd.read_csv('train_data.csv')
        sample_data = train_data.head(10)
        
        # Create temporary CSV
        temp_csv = 'temp_test_data.csv'
        sample_data.to_csv(temp_csv, index=False)
        
        # Load dataset
        ds = load_dataset("csv", data_files=temp_csv)
        
        # Test formatting function
        def format_example(example):
            example["text"] = (
                "Classify the stance of the following tweet as "
                "in-favor, against, or neutral-or-unclear.\n\nTweet:\n"
                f"{example['tweet_clean']}\n\nAnswer:"
            )
            example["label_text"] = example["label_true"]
            return example
        
        # Format examples
        ds = ds.map(format_example)
        print("✅ Data formatting successful")
        
        # Test tokenization
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        def tokenize_function(examples):
            model_inputs = tokenizer(
                examples["text"], 
                truncation=True, 
                padding="max_length", 
                max_length=256
            )
            labels = tokenizer(
                examples["label_text"], 
                truncation=True, 
                padding="max_length", 
                max_length=16
            )
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs
        
        # Tokenize dataset
        ds_tok = ds.map(tokenize_function, batched=True, remove_columns=ds["train"].column_names)
        print("✅ Tokenization successful")
        
        # Clean up
        if os.path.exists(temp_csv):
            os.remove(temp_csv)
        
        return True
        
    except Exception as e:
        print(f"❌ Error in data processing: {e}")
        return False

def test_metrics_function():
    """Test the metrics calculation function."""
    print("\nTesting metrics calculation...")
    
    try:
        # Mock predictions and labels
        predictions = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]])
        labels = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]])
        
        # Mock tokenizer
        class MockTokenizer:
            def batch_decode(self, tokens, skip_special_tokens=True):
                return ["in-favor", "against", "neutral-or-unclear"]
        
        tokenizer = MockTokenizer()
        
        # Test metrics function
        def macro_f1_metrics(eval_pred):
            predictions, labels = eval_pred
            
            # Decode predictions
            decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
            
            # Replace -100 in the labels as we can't decode them
            labels = np.where(labels != -100, labels, 0)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            # Extract labels from text
            pred_labels = []
            true_labels = []
            
            label_mapping = {
                'in-favor': 'in-favor',
                'against': 'against', 
                'neutral-or-unclear': 'neutral-or-unclear'
            }
            
            for pred, label in zip(decoded_preds, decoded_labels):
                pred_clean = pred.strip().lower()
                label_clean = label.strip().lower()
                
                # Map predictions to labels
                pred_label = None
                for key, value in label_mapping.items():
                    if key.lower() in pred_clean:
                        pred_label = value
                        break
                
                if pred_label is None:
                    pred_label = 'neutral-or-unclear'  # default
                
                true_label = None
                for key, value in label_mapping.items():
                    if key.lower() in label_clean:
                        true_label = value
                        break
                
                if true_label is None:
                    true_label = 'neutral-or-unclear'  # default
                
                pred_labels.append(pred_label)
                true_labels.append(true_label)
            
            # Calculate metrics
            from sklearn.metrics import f1_score
            f1_macro = f1_score(true_labels, pred_labels, average='macro')
            f1_weighted = f1_score(true_labels, pred_labels, average='weighted')
            
            # Calculate accuracy
            accuracy = sum(1 for p, t in zip(pred_labels, true_labels) if p == t) / len(pred_labels)
            
            return {
                'eval_macro_f1': f1_macro,
                'eval_weighted_f1': f1_weighted,
                'eval_accuracy': accuracy
            }
        
        # Test the function
        metrics = macro_f1_metrics((predictions, labels))
        print("✅ Metrics calculation successful")
        print(f"  Mock metrics: {metrics}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in metrics calculation: {e}")
        return False

def main():
    """Run all tests."""
    print("="*60)
    print("TRAINING SETUP TEST")
    print("="*60)
    
    tests = [
        ("Data Loading", test_data_loading),
        ("Model Loading", test_model_loading),
        ("Data Processing", test_data_processing),
        ("Metrics Calculation", test_metrics_function)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    print("\n" + "="*60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print("="*60)
    
    if passed == total:
        print("🎉 All tests passed! Training setup is ready.")
        print("\nYou can now run the training with:")
        print("python train_flan_t5.py")
    else:
        print("⚠️  Some tests failed. Please fix the issues before training.")
    
    return passed == total

if __name__ == "__main__":
    main() 