import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# Load intents.json
with open('intents.json', 'r') as file:
    intents = json.load(file)

# Prepare dataset
data = []
labels = []
label_to_id = {}
id_to_label = {}

for intent in intents['intents']:
    label = intent['tag']
    if label not in label_to_id:
        label_id = len(label_to_id)
        label_to_id[label] = label_id
        id_to_label[label_id] = label
    for pattern in intent['patterns']:
        data.append(pattern)
        labels.append(label_to_id[label])

# Convert to Hugging Face Dataset
dataset = Dataset.from_dict({"text": data, "label": labels})

# Tokenizer and Model
token = "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"
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small", token=token)
model = AutoModelForSequenceClassification.from_pretrained("google/flan-t5-small", num_labels=len(label_to_id), token=token)

# Tokenize dataset
def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True)

dataset = dataset.map(tokenize, batched=True)

# Format dataset for Trainer
dataset = dataset.rename_column("label", "labels")
dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# Split dataset into train and validation sets
train_test_split = dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']

# Training Arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=40,
    per_device_train_batch_size=20,
    per_device_eval_batch_size=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir='./logs',
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)

# Compute metrics
def compute_metrics(p):
    preds = torch.argmax(torch.tensor(p.predictions[0]), dim=-1)  # Convert to tensor
    precision, recall, f1, _ = precision_recall_fscore_support(p.label_ids, preds, average='weighted')
    acc = accuracy_score(p.label_ids, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Evaluate the model
metrics = trainer.evaluate()
print(metrics)