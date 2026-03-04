from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import evaluate
import numpy as np

# charger les données via HuggingFace
dataset = load_dataset("multi_nli")

# pour aller plus vite au début, on prend un petit bout (ex: 10000 pour train, 2000 pour dev)
train_dataset = dataset['train'].shuffle(seed=42).select(range(10000))
eval_dataset = dataset['validation_matched'].shuffle(seed=42).select(range(2000))

# tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    # tokenize les deux phrases ensemble
    return tokenizer(examples["premise"], examples["hypothesis"], padding="max_length", truncation=True, max_length=128)

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_eval = eval_dataset.map(tokenize_function, batched=True)

# modèle
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

# métrique (Accuracy)
metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# entraînement (Trainer)
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    compute_metrics=compute_metrics,
)

# lancer l'entraînement
# trainer.train()