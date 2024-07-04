import sys
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, DataCollatorForLanguageModeling, Trainer
from datasets import load_dataset, Dataset
import torch

#import os
#os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Function to concatenate the "system", "user", and "assistant" fields into a single text
def concatenate_fields(examples):
    concatenated_examples = []
    for system, user, assistant in zip(examples['system'], examples['user'], examples['assistant']):
        concatenated_text = f"System: {system}\nUser: {user}\nAssistant: {assistant}"
        concatenated_examples.append(concatenated_text)
    return {"text": concatenated_examples}

# Tokenize the concatenated text
def tokenize_function(examples, max_length=512):
    tokens = tokenizer(examples['text'], padding="max_length", truncation=True, max_length=max_length)
    return tokens

# Function to move the batch to the specified device
#def collate_fn(batch, device):
#    for k, v in batch.items():
#        batch[k] = v.to(device)
#    return batch

model_name="instructlab/merlinite-7b-lab"
# Verifica la disponibilità della GPU MPS
device = "cpu"
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = "mps"
torch_device = torch.device(device)
print(f"* Using device: {device}")

print(f"* Set the default device for PyTorch to {torch_device}")
torch.set_default_device(torch_device)

# Prepare the tokenizer for the input model. AutoTokenizer will load the correct tokenizer for the input model.
print(f"* Load the tokenizer for {model_name}.")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
# Load the model in input. If the models doesn't exist locally it is downloaded. AutoModelForCausalLM will load the correct model
# for the input model name.
print(f"* Load the model for {model_name} to the {torch_device} device.")
model = AutoModelForCausalLM.from_pretrained(model_name, device_map=torch_device, trust_remote_code=True)
print(f"Model is on the device: {next(model.parameters()).device}")

# Load the dataset
dataset_path = 'train_merlinite_7b.jsonl'
print(f"* Load the training dataset {dataset_path}.")
dataset = load_dataset("json", data_files=dataset_path)

# Prepare the input dataset in the following way:
# - create a column text with the concatenation of system, user, assistant columns.
# - tokenize the text column
# - remove the system, user, assistant columns
print(f"* Tokenize the input dataset.")
dataset = dataset.map(concatenate_fields, batched=True)
tokenized_datasets = dataset.map(lambda examples: tokenize_function(examples, 512), batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["system", "user", "assistant", "text"])
print(f"* Convert the dataset in PyTorch format.")
tokenized_datasets.set_format("torch")

# Assuming the dataset has a "train" split
train_dataset = tokenized_datasets["train"]

# Define training arguments
print(f"* Prepare for the training of the {model_name} model.")
training_args = TrainingArguments(
    output_dir="./results",           # Directory di output per i modelli e i log
    eval_strategy="epoch",            # Strategia di valutazione (ad ogni epoca)
    learning_rate=2e-5,               # Tasso di apprendimento
    per_device_train_batch_size=4,    # Batch size per dispositivo durante l'addestramento
    per_device_eval_batch_size=4,     # Batch size per dispositivo durante la valutazione
    num_train_epochs=3,               # Numero di epoche di addestramento
    weight_decay=0.01,                # Decadimento del peso
    save_total_limit=2,               # Numero massimo di checkpoint da conservare
    save_steps=10_000,                # Frequenza di salvataggio dei checkpoint
    logging_dir='./logs',             # Directory per i log
    use_mps_device=True,
)

# Initialize the DataCollator
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# Initialize the SFTTrainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Start training
print(f"* Train the {model_name} model.")
trainer.train()
