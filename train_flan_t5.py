#!/usr/bin/env python3
"""
Training script for Flan-T5-large model on tweet classification task.

This script fine-tunes the Flan-T5-large model using LoRA (Low-Rank Adaptation)
with weighted loss to handle class imbalance.
"""

import pandas as pd
import numpy as np
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType
import torch
from sklearn.metrics import classification_report, f1_score
import json
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class WeightedTrainer:
    """
    Custom trainer with weighted loss to handle class imbalance.
    """
    
    def __init__(self, model, args, train_dataset, eval_dataset, tokenizer, 
                 compute_metrics, class_weights):
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.compute_metrics = compute_metrics
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        
    def train(self):
        """
        Train the model with weighted loss.
        """
        from transformers import Trainer
        
        # Create a custom trainer with weighted loss
        trainer = Trainer(
            model=self.model,
            args=self.args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
            data_collator=DataCollatorForSeq2Seq(
                tokenizer=self.tokenizer,
                padding=True,
                return_tensors="pt"
            )
        )
        
        # Train the model
        trainer.train()
        
        # Save the model
        trainer.save_model()
        
        return trainer

def format_example(example):
    """
    Format the example for training.
    
    Args:
        example (dict): Input example
        
    Returns:
        dict: Formatted example
    """
    example["text"] = (
        "Classify the stance of the following tweet as "
        "in-favor, against, or neutral-or-unclear.\n\nTweet:\n"
        f"{example['tweet_clean']}\n\nAnswer:"
    )
    example["label_text"] = example["label_true"]
    return example

def tokenize_function(examples):
    """
    Tokenize the examples for training.
    
    Args:
        examples (dict): Input examples
        
    Returns:
        dict: Tokenized examples
    """
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

def macro_f1_metrics(eval_pred):
    """
    Compute macro F1 score for evaluation.
    
    Args:
        eval_pred (tuple): Predictions and labels
        
    Returns:
        dict: Metrics
    """
    predictions, labels = eval_pred
    
    # Decode predictions
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # Replace -100 in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
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
    f1_macro = f1_score(true_labels, pred_labels, average='macro')
    f1_weighted = f1_score(true_labels, pred_labels, average='weighted')
    
    # Calculate accuracy
    accuracy = sum(1 for p, t in zip(pred_labels, true_labels) if p == t) / len(pred_labels)
    
    return {
        'eval_macro_f1': f1_macro,
        'eval_weighted_f1': f1_weighted,
        'eval_accuracy': accuracy
    }

def prepare_dataset():
    """
    Prepare the dataset for training.
    
    Returns:
        DatasetDict: Prepared dataset
    """
    print("Loading and preparing dataset...")
    
    # Load the prepared train data
    train_data = pd.read_csv('train_data.csv')
    
    # Create a temporary CSV for the datasets library
    temp_csv = 'temp_train_data.csv'
    train_data.to_csv(temp_csv, index=False)
    
    # Load dataset
    ds = load_dataset("csv", data_files=temp_csv)
    
    # Format examples
    ds = ds.map(format_example)
    
    # Split into train and validation
    ds = ds["train"].train_test_split(
        test_size=0.2, 
        stratify_by_column="label_true", 
        seed=42
    )
    
    # Further split train into train and eval
    train_eval_split = ds["train"].train_test_split(
        test_size=0.125, 
        stratify_by_column="label_true", 
        seed=42
    )
    
    ds = DatasetDict({
        "train": train_eval_split["train"],
        "eval": train_eval_split["test"],
        "test": ds["test"]
    })
    
    print(f"Dataset splits:")
    print(f"  Train: {len(ds['train'])} samples")
    print(f"  Eval: {len(ds['eval'])} samples")
    print(f"  Test: {len(ds['test'])} samples")
    
    # Tokenize the dataset
    ds_tok = ds.map(tokenize_function, batched=True, remove_columns=ds["train"].column_names)
    
    # Clean up temporary file
    if os.path.exists(temp_csv):
        os.remove(temp_csv)
    
    return ds_tok

def setup_model_and_tokenizer():
    """
    Setup the model and tokenizer.
    
    Returns:
        tuple: (model, tokenizer)
    """
    global tokenizer
    
    model_name = "google/flan-t5-large"
    print(f"Loading model and tokenizer: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
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
    
    print(f"Model loaded successfully")
    print(f"Trainable parameters: {model.print_trainable_parameters()}")
    
    return model, tokenizer

def setup_training_args():
    """
    Setup training arguments.
    
    Returns:
        TrainingArguments: Training arguments
    """
    return TrainingArguments(
        output_dir="flan_t5_stance_classifier",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        warmup_steps=100,
        logging_steps=50,
        save_total_limit=2,
        dataloader_pin_memory=False,
        fp16=torch.cuda.is_available(),
        report_to=None,  # Disable wandb
    )

def train_model():
    """
    Main training function.
    """
    print("="*60)
    print("FLAN-T5-LARGE TRAINING")
    print("="*60)
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer()
    
    # Prepare dataset
    ds_tok = prepare_dataset()
    
    # Setup training arguments
    args = setup_training_args()
    
    # Calculate class weights based on training data distribution
    train_labels = ds_tok["train"]["label_true"]
    label_counts = pd.Series(train_labels).value_counts()
    total_samples = len(train_labels)
    
    # Calculate inverse frequency weights
    class_weights = []
    for label in ['in-favor', 'against', 'neutral-or-unclear']:
        if label in label_counts:
            weight = total_samples / (len(label_counts) * label_counts[label])
            class_weights.append(weight)
        else:
            class_weights.append(1.0)
    
    print(f"Class weights: {class_weights}")
    print(f"Label distribution in training set:")
    for label, count in label_counts.items():
        print(f"  {label}: {count} samples")
    
    # Create trainer
    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=ds_tok["train"],
        eval_dataset=ds_tok["eval"],
        tokenizer=tokenizer,
        compute_metrics=macro_f1_metrics,
        class_weights=class_weights
    )
    
    # Train the model
    print("\nStarting training...")
    trainer.train()
    
    # Save tokenizer
    tokenizer.save_pretrained("flan_t5_stance_classifier")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED!")
    print("="*60)
    print("Model saved to: flan_t5_stance_classifier/")
    print("You can now use the trained model for evaluation.")

def main():
    """
    Main function to run training.
    """
    # Check if training data exists
    if not os.path.exists('train_data.csv'):
        print("Error: train_data.csv not found!")
        print("Please run the data preparation script first:")
        print("python data_preparation.py")
        return
    
    # Start training
    train_model()

if __name__ == "__main__":
    main() 