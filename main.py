from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
import torch
from trl import SFTTrainer

# Function to concatenate the "system", "user", and "assistant" fields into a single text
def concatenate_fields(examples):
    concatenated_examples = []
    for system, user, assistant in zip(examples['system'], examples['user'], examples['assistant']):
        concatenated_text = f"System: {system}\nUser: {user}\nAssistant: {assistant}"
        concatenated_examples.append(concatenated_text)
    return {"text": concatenated_examples}

# Tokenize the concatenated text
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

model_name="instructlab/merlinite-7b-lab"
# Verifica la disponibilità della GPU MPS
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = "mps"
    device = torch.device(device)
print(f"* Using device: {device}")

# Prepare the tokenizer for the input model. AutoTokenizer will load the correct tokenizer for the input model.
print(f"* Load the tokenizer for {model_name}.")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
# Load the model in input. If the models doesn't exist locally it is downloaded. AutoModelForCausalLM will load the correct model
# for the input model name.
print(f"* Load the model for {model_name}.")
model = AutoModelForCausalLM.from_pretrained(model_name)

# Move model on the GPU if available
print(f"* Move the model {model_name} to the {device} device.")
if model.device != device:
    model.to(device)

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
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Convert the tokenized dataset in PyTorch format.
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
)

# Initialize the DataCollator
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# Initialize the SFTTrainer
trainer = SFTTrainer(
    model=model,                        # Modello da addestrare
    args=training_args,                 # Argomenti di addestramento
    train_dataset=train_dataset,        # Dataset di addestramento
    tokenizer=tokenizer,                # Tokenizer
    data_collator=data_collator         # DataCollator per il padding
    # Sposta automaticamente i batch di input sul dispositivo appropriato
    #data_collator=lambda data: {k: torch.stack(v).to(device) if isinstance(v, list) else v.to(device) for k, v in data.items()}
    #                          if isinstance(data, dict) else (torch.stack(data).to(device) if isinstance(data, list) else data.to(device))
    #                          if not isinstance(data, str) else data
)

# Start training
print(f"* Train the {model_name} model.")
trainer.train()
