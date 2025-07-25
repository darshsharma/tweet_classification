import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from tqdm import tqdm

BATCH_SIZE = 16  # Adjust as needed

# 1. Load fine-tuned model and tokenizer
model_dir = "./trainer_saved"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
model.eval()

# 2. Load CSV
df = pd.read_csv("Q2_20230202_majority.csv")

# 3. Prompt template
prompt_template = (
    'What is the stance of the following tweet with respect to COVID-19 vaccine? '
    'Here is the tweet. "{tweet}"  Please use exactly one word from the following 3 categories to label it: '
    '"in-favor", "against", "neutral-or-unclear".'
)

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

preds = []
tweets = df["tweet"].tolist()
for tweet_batch in tqdm(list(batch(tweets, BATCH_SIZE)), desc="Predicting"):
    prompts = [prompt_template.format(tweet=t) for t in tweet_batch]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=5
        )
    batch_preds = [tokenizer.decode(o, skip_special_tokens=True).strip() for o in outputs]
    # Clean up predictions
    batch_preds = [
        p if p in ["in-favor", "against", "neutral-or-unclear"] else "neutral-or-unclear"
        for p in batch_preds
    ]
    preds.extend(batch_preds)

df["label_pred"] = preds
df.to_csv("Q2_20230202_majority_with_preds.csv", index=False)
print("Done! Predictions saved to Q2_20230202_majority_with_preds.csv")