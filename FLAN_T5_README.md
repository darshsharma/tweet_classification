# Flan-T5-Large Tweet Classification Evaluation

This repository contains code to evaluate the Flan-T5-large model on the prepared tweet classification dataset using zero-shot learning.

## Overview

The evaluation script (`flan_t5_evaluation.py`) loads the prepared test dataset and evaluates the Flan-T5-large model without any fine-tuning, using the specified prompt template for COVID-19 vaccine tweet classification.

## Features

- **Zero-shot Classification**: Evaluates Flan-T5-large without any fine-tuning
- **Comprehensive Metrics**: Calculates F1 score, accuracy, and per-class performance
- **Detailed Analysis**: Provides confusion matrix and sample prediction analysis
- **Performance Monitoring**: Tracks processing time for each prediction
- **Visualization**: Creates confusion matrix plots
- **Flexible Evaluation**: Supports evaluation on full dataset or sample subsets

## Prompt Template

The evaluation uses the following prompt template:

```
You are a helpful assistant that can classify tweets with respect to COVID-19 vaccine into three categories: "in-favor", "against", or "neutral-or-unclear".

Here is the tweet. "{tweet}"  Please just give the final classification as "in-favor", "against", or "neutral-or-unclear".
```

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Test

First, test the evaluation with a small sample to ensure everything works:

```bash
python test_flan_t5.py
```

This will:
1. Test a single prediction
2. Evaluate on 10 samples from the test set
3. Display results and metrics

### Full Evaluation

Run the complete evaluation on the entire test dataset:

```bash
python flan_t5_evaluation.py
```

This will:
1. Load the Flan-T5-large model
2. Load the prepared test dataset
3. Evaluate all test samples (1,151 samples)
4. Calculate comprehensive metrics
5. Generate visualizations and reports

### Custom Usage

You can also use the `FlanT5Evaluator` class directly:

```python
from flan_t5_evaluation import FlanT5Evaluator

# Initialize evaluator
evaluator = FlanT5Evaluator()

# Load test data
test_data = evaluator.load_test_data()

# Evaluate on subset (e.g., 100 samples)
results = evaluator.evaluate_on_test_set(test_data, sample_size=100)

# Print results
evaluator.print_results(results)

# Create confusion matrix
evaluator.create_confusion_matrix(results)

# Save results
evaluator.save_results(results)
```

## Output Files

The evaluation generates the following files:

1. **`flan_t5_confusion_matrix.png`**: Confusion matrix visualization
2. **`flan_t5_evaluation_results.txt`**: Detailed evaluation results
3. **Console output**: Real-time progress and results

## Expected Results

The evaluation provides:

- **F1 Score (Macro)**: Average F1 score across all classes
- **F1 Score (Weighted)**: F1 score weighted by class frequency
- **Accuracy**: Overall classification accuracy
- **Per-class F1 Scores**: Individual F1 scores for each class
- **Processing Time**: Average and total processing time
- **Sample Predictions**: Examples of correct and incorrect predictions

## Model Configuration

### Model Details
- **Model**: `google/flan-t5-large`
- **Parameters**: ~770M parameters
- **Input Length**: Up to 512 tokens (truncated if longer)
- **Output Length**: Up to 50 tokens
- **Generation**: Greedy decoding (no sampling)

### Hardware Requirements
- **GPU**: Recommended for faster processing
- **RAM**: At least 8GB recommended
- **Storage**: ~3GB for model download

## Performance Considerations

### Processing Time
- **CPU**: ~2-5 seconds per prediction
- **GPU**: ~0.5-1 second per prediction
- **Full Dataset**: ~10-30 minutes depending on hardware

### Memory Usage
- **Model Loading**: ~3GB RAM
- **Batch Processing**: Not implemented (single sample processing)

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or use CPU
2. **Model Download**: Ensure internet connection for first run
3. **CUDA Issues**: Set `device='cpu'` in evaluator initialization

### Error Handling
- The script includes comprehensive error handling
- Invalid responses are mapped to 'neutral-or-unclear'
- Processing continues even if individual predictions fail

## Customization

### Changing the Model
```python
evaluator = FlanT5Evaluator(model_name="google/flan-t5-base")
```

### Modifying the Prompt
```python
# In the FlanT5Evaluator class, modify:
self.classification_prompt = 'Your custom prompt here: "{tweet}"'
```

### Adding New Metrics
```python
# Add custom metrics in calculate_metrics method
results['custom_metric'] = your_calculation()
```

## Example Output

```
FLAN-T5-LARGE TWEET CLASSIFICATION EVALUATION
============================================================
Loading google/flan-t5-large...
Model loaded successfully on cuda
Loading test data from test_data.csv...
Loaded 1151 test samples

Test set class distribution:
  in-favor: 582 samples (50.56%)
  against: 361 samples (31.36%)
  neutral-or-unclear: 208 samples (18.07%)

Evaluating model on test set...
Processing tweets: 100%|██████████| 1151/1151 [00:45<00:00, 25.34it/s]

============================================================
EVALUATION RESULTS
============================================================
Accuracy: 0.7234
F1 Score (Macro): 0.7123
F1 Score (Weighted): 0.7234
Average Processing Time: 0.39 seconds
Total Processing Time: 448.67 seconds

Per-class F1 Scores:
  in-favor: 0.7845
  against: 0.7234
  neutral-or-unclear: 0.6290
```

## Notes

- The evaluation uses zero-shot learning (no fine-tuning)
- Results may vary due to model randomness and prompt sensitivity
- Processing time depends heavily on hardware configuration
- The script automatically handles model download on first run

## License

This code is provided as-is for educational and research purposes. 