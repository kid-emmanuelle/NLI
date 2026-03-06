from datasets import load_dataset, DatasetDict, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import KFold
import numpy as np
import evaluate
from collections import defaultdict
import json
import torch
import warnings

# Supprime les warnings inutiles
warnings.filterwarnings('ignore', message='.*pin_memory.*')
warnings.filterwarnings('ignore', message='.*accelerator.*')

# Vérifier la disponibilité du GPU
use_cuda = torch.cuda.is_available()
print(f"GPU disponible: {use_cuda}")
if use_cuda:
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# charger les données via HuggingFace
dataset = load_dataset("multi_nli")

# utiliser un subset pour plus de rapidité
# prendre train et validation_matched
train_data = dataset['train'].shuffle(seed=42).select(range(10000))
print(f"Données d'entraînement: {len(train_data)} exemples")

# tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(
        examples["premise"], 
        examples["hypothesis"], 
        padding="max_length", 
        truncation=True, 
        max_length=128
    )

# K-Fold configuration
n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# métrique
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Stocker les résultats de chaque fold
fold_results = defaultdict(list)
fold_number = 0

# Préparer les indices pour k-fold
indices = np.arange(len(train_data))
labels_array = np.array(train_data["label"])

# K-Fold cross-validation
for train_idx, eval_idx in kfold.split(indices, labels_array):
    fold_number += 1
    print(f"\n{'='*50}")
    print(f"FOLD {fold_number}/{n_splits}")
    print(f"{'='*50}")
    
    # Créer train et eval datasets pour ce fold
    fold_train_data = train_data.select(train_idx)
    fold_eval_data = train_data.select(eval_idx)
    
    print(f"Train size: {len(fold_train_data)}, Eval size: {len(fold_eval_data)}")
    
    # Tokenization
    tokenized_train = fold_train_data.map(tokenize_function, batched=True)
    tokenized_eval = fold_eval_data.map(tokenize_function, batched=True)
    
    # Modèle frais pour chaque fold
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    
    # Arguments d'entraînement
    training_args = TrainingArguments(
        output_dir=f"./results/fold_{fold_number}",
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=2,
        logging_steps=100,
        save_strategy="no",
        dataloader_pin_memory=False  # ne pas sauvegarder les checkpoints intermédiaires
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        compute_metrics=compute_metrics,
    )
    
    # Entraîner
    trainer.train()
    
    # Évaluation finale
    eval_results = trainer.evaluate()
    fold_results['accuracy'].append(eval_results['eval_accuracy'])
    fold_results['loss'].append(eval_results['eval_loss'])
    
    print(f"Fold {fold_number} - Accuracy: {eval_results['eval_accuracy']:.4f}, Loss: {eval_results['eval_loss']:.4f}")

# Résultats finaux
print(f"\n{'='*50}")
print("RÉSULTATS FINAUX K-FOLD")
print(f"{'='*50}")
print(f"Accuracy moyenne: {np.mean(fold_results['accuracy']):.4f} (+/- {np.std(fold_results['accuracy']):.4f})")
print(f"Loss moyenne: {np.mean(fold_results['loss']):.4f} (+/- {np.std(fold_results['loss']):.4f})")
print(f"\nDétails par fold:")
for i, (acc, loss) in enumerate(zip(fold_results['accuracy'], fold_results['loss']), 1):
    print(f"  Fold {i}: Accuracy={acc:.4f}, Loss={loss:.4f}")

# Sauvegarder les résultats
results_summary = {
    'n_splits': n_splits,
    'accuracy_mean': float(np.mean(fold_results['accuracy'])),
    'accuracy_std': float(np.std(fold_results['accuracy'])),
    'loss_mean': float(np.mean(fold_results['loss'])),
    'loss_std': float(np.std(fold_results['loss'])),
    'fold_details': {
        f'fold_{i+1}': {'accuracy': float(acc), 'loss': float(loss)}
        for i, (acc, loss) in enumerate(zip(fold_results['accuracy'], fold_results['loss']))
    }
}

with open('./results/kfold_results.json', 'w') as f:
    json.dump(results_summary, f, indent=2)

print("\nRésultats sauvegardés dans ./results/kfold_results.json")
