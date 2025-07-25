import pandas as pd
from datasets import Dataset
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

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
    tokenizer=tokenizer,
    peft_config=lora_config,
    dataset_text_field="text",
    max_seq_length=512
)

trainer.train()
trainer.save_model("./trainer_saved")
print("Fine-tuned model saved to ./trainer_saved")