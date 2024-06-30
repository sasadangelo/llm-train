from transformers import LlamaTokenizer, LlamaForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
import torch

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

# Verifica la disponibilità della GPU MPS
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Prepare the tokenizer for the input model
tokenizer = LlamaTokenizer.from_pretrained('instructlab/merlinite-7b-lab')
# Load the model in input. If the models doesn't exist locally it is downloaded
model = LlamaForCausalLM.from_pretrained('instructlab/merlinite-7b-lab')

# Move model on the GPU if available
model.to(device)

# Load the dataset
dataset_path = '/Users/sasadangelo/github.com/sasadangelo/llm-train/train_merlinite_7b.jsonl'
dataset = load_dataset("json", data_files=dataset_path)

# Apply the function to concatenate the fields
dataset = dataset.map(concatenate_fields, batched=True)

# Visualizza le prime righe del dataset concatenato
#print(dataset['train'].head())

#for i in range(dataset['train'].num_rows):
#    print(dataset['train'][i])

# Tokenize the concatenated text
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Remove the original text columns: "system", "user", "assistant" and the temporary "text"
tokenized_datasets = tokenized_datasets.remove_columns(["system", "user", "assistant", "text"])
tokenized_datasets.set_format("torch")

# Assuming the dataset has a "train" split
train_dataset = tokenized_datasets["train"]

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",           # Directory di output per i modelli e i log
    use_mps_device=True,
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

# Initialize the Trainer
trainer = Trainer(
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
trainer.train()
