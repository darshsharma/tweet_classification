import pandas as pd
from datasets import Dataset
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm

df = pd.read_csv("/content/tweet_classification/Q2_20230202_majority.csv")
df = df[df["label_true"].notnull()]  # Only use labeled data

train_df, val_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df["label_true"])

train_df.to_csv("train.csv", index=False)
val_df.to_csv("val.csv", index=False)
print(f"Train size: {len(train_df)}, Validation size: {len(val_df)}")



# Load train and val
train_df = pd.read_csv("train.csv")
val_df = pd.read_csv("val.csv")

prompt_template = (
    'You are a helpful assistant that can classify tweets with respect to COVID-19 vaccine into three categories: "in-favor", "against", or "neutral-or-unclear".'
    'Here is the tweet. "{tweet}"  Please just give the final classification as "in-favor", "against", or "neutral-or-unclear".')

for df in [train_df, val_df]:
    df["text"] = df["tweet"].apply(lambda t: prompt_template.format(tweet=t))
    df["labels"] = df["label_true"]

train_dataset = Dataset.from_pandas(train_df[["text", "labels"]])
val_dataset = Dataset.from_pandas(val_df[["text", "labels"]])

def preprocess_function(examples):
    model_inputs = tokenizer(
        examples["text"], max_length=512, truncation=True, padding="max_length"
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["labels"], max_length=10, truncation=True, padding="max_length"
        )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

train_dataset = train_dataset.map(preprocess_function, batched=True)
val_dataset = val_dataset.map(preprocess_function, batched=True)

# 2. Load model and tokenizer
model_id = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_id)
base_model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

# 3. LoRA config and model
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_2_SEQ_LM"
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()

# 4. Training arguments
training_args = TrainingArguments(
    output_dir="./trainer_saved",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=torch.cuda.is_available(),
    logging_steps=20,
    save_strategy="epoch",
)

# 5. Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    args=training_args,
    peft_config=lora_config
)

trainer.train()
trainer.save_model("./trainer_saved")
print("Fine-tuned model saved to ./trainer_saved")




BATCH_SIZE = 16
model_dir = "./trainer_saved"  # or your model output dir

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
model.eval()

val_df = pd.read_csv("val.csv")
prompt_template = (
    'You are a helpful assistant that can classify tweets with respect to COVID-19 vaccine into three categories: "in-favor", "against", or "neutral-or-unclear".'
    'Here is the tweet. "{tweet}"  Please just give the final classification as "in-favor", "against", or "neutral-or-unclear".')

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

preds = []
tweets = val_df["tweet"].tolist()
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
    batch_preds = [
        p if p in ["in-favor", "against", "neutral-or-unclear"] else "neutral-or-unclear"
        for p in batch_preds
    ]
    preds.extend(batch_preds)

val_df["label_pred"] = preds
val_df.to_csv("val_with_preds.csv", index=False)

# Evaluation
print(classification_report(val_df["label_true"], val_df["label_pred"], digits=4))