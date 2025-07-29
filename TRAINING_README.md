# Flan-T5-Large Training and Evaluation

This document describes the training and evaluation process for fine-tuning the Flan-T5-large model on the COVID-19 vaccine tweet classification task.

## Overview

The training process uses LoRA (Low-Rank Adaptation) to efficiently fine-tune the Flan-T5-large model for the tweet classification task. This approach allows us to train the model with limited computational resources while maintaining good performance.

## Files

### Training Scripts
- `train_flan_t5.py` - Main training script using LoRA fine-tuning
- `evaluate_trained_model.py` - Evaluation script for the trained model

### Data Files
- `train_data.csv` - Prepared training dataset (80% of original data)
- `test_data.csv` - Prepared test dataset (20% of original data)

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Training Process

### 1. Data Preparation
First, ensure you have the prepared datasets:
```bash
python data_preparation.py
```

This creates:
- `train_data.csv` - Training dataset with 4,600 samples
- `test_data.csv` - Test dataset with 1,151 samples

### 2. Training the Model
Run the training script:
```bash
python train_flan_t5.py
```

**Training Configuration:**
- **Model**: google/flan-t5-large
- **Method**: LoRA (Low-Rank Adaptation)
- **LoRA Config**: r=8, alpha=16, dropout=0.05
- **Target Modules**: ["q", "v"] (query and value matrices)
- **Batch Size**: 4 (per device)
- **Learning Rate**: 2e-4
- **Epochs**: 3
- **Evaluation**: Every epoch
- **Metric**: Macro F1 score

**Class Weights:**
The training automatically calculates class weights to handle imbalanced data:
- Weights are calculated based on inverse frequency
- Helps improve performance on minority classes

### 3. Training Output
The training process will:
- Load the Flan-T5-large model
- Apply LoRA adapters
- Train for 3 epochs
- Save the best model based on validation F1 score
- Save the model to `flan_t5_stance_classifier/`

## Evaluation

### 1. Evaluate Trained Model
After training, evaluate the model:
```bash
python evaluate_trained_model.py
```

This script will:
- Load the trained model
- Evaluate on the test dataset
- Compare with zero-shot baseline
- Generate confusion matrix
- Save detailed results

### 2. Comparison with Baseline
The evaluation automatically compares the trained model with the zero-shot baseline:
- **Zero-shot**: F1 (Macro) = 0.4799, Accuracy = 0.5882
- **Trained**: Expected improvement in performance

## Model Architecture

### LoRA Configuration
```python
peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    r=8,                    # Rank of the low-rank matrices
    lora_alpha=16,          # Scaling factor
    lora_dropout=0.05,      # Dropout rate
    target_modules=["q", "v"]  # Target attention modules
)
```

### Training Arguments
```python
TrainingArguments(
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
    fp16=torch.cuda.is_available(),
)
```

## Expected Results

### Performance Metrics
The trained model is expected to show improvements over the zero-shot baseline:

| Metric | Zero-shot | Trained (Expected) |
|--------|-----------|-------------------|
| F1 (Macro) | 0.4799 | >0.60 |
| F1 (Weighted) | 0.5568 | >0.65 |
| Accuracy | 0.5882 | >0.70 |

### Per-class Performance
Expected improvements across all classes:
- **in-favor**: F1 > 0.70
- **against**: F1 > 0.60  
- **neutral-or-unclear**: F1 > 0.40

## Files Generated

### Training Output
- `flan_t5_stance_classifier/` - Trained model directory
  - `adapter_config.json` - LoRA configuration
  - `adapter_model.bin` - LoRA weights
  - `tokenizer.json` - Tokenizer files
  - `special_tokens_map.json`
  - `tokenizer_config.json`

### Evaluation Output
- `trained_model_confusion_matrix.png` - Confusion matrix visualization
- `trained_model_evaluation_results.txt` - Detailed evaluation results

## Usage Examples

### Load Trained Model
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("flan_t5_stance_classifier")

# Load base model
base_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")

# Load trained model
model = PeftModel.from_pretrained(base_model, "flan_t5_stance_classifier")
```

### Make Predictions
```python
def predict_tweet(tweet_text):
    prompt = f"Classify the stance of the following tweet as in-favor, against, or neutral-or-unclear.\n\nTweet:\n{tweet_text}\n\nAnswer:"
    
    inputs = tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True)
    outputs = model.generate(**inputs, max_length=16)
    
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return prediction.strip()
```

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or use gradient accumulation
2. **Slow Training**: Use GPU if available, reduce model size
3. **Poor Performance**: Increase epochs, adjust learning rate, check data quality

### Performance Tips

1. **GPU Usage**: Training is much faster on GPU
2. **Batch Size**: Adjust based on available memory
3. **Mixed Precision**: Use fp16 for faster training
4. **Gradient Checkpointing**: Enable for memory efficiency

## Comparison with Zero-shot

The training process addresses several limitations of zero-shot classification:

1. **Better Understanding**: Model learns task-specific patterns
2. **Improved Consistency**: More reliable predictions
3. **Class Balance**: Handles imbalanced data better
4. **Domain Adaptation**: Adapts to tweet-specific language

## Next Steps

After training and evaluation:

1. **Hyperparameter Tuning**: Experiment with different LoRA configurations
2. **Data Augmentation**: Increase training data diversity
3. **Ensemble Methods**: Combine multiple models
4. **Production Deployment**: Optimize for inference speed

## References

- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [PEFT Library](https://github.com/huggingface/peft)
- [Flan-T5 Paper](https://arxiv.org/abs/2210.11416)
- [Transformers Documentation](https://huggingface.co/docs/transformers/) 