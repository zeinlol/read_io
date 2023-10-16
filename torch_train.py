import pandas as pd
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer

from classes import TextClassifier, CustomDataset
from constants import TRAINING_LOOPS, BATCH_SIZE

data = pd.read_csv("train.csv")

# Split your data into training and validation sets
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

# Initialize the RoBERTa tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

train_dataset = CustomDataset(train_data["excerpt"], train_data["target"], tokenizer)
val_dataset = CustomDataset(val_data["excerpt"], val_data["target"], tokenizer)

# Define a DataLoader for batch processing
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Initialize your model and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'ML will use {device}')
model_to_train = TextClassifier().to(device)
optimizer = optim.AdamW(model_to_train.parameters(), lr=2e-5)

# Define your loss function
criterion = nn.MSELoss()

# Training loop
for epoch in range(TRAINING_LOOPS):
    model_to_train.train()
    try:
        for batch in train_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            squeezed = model_to_train(input_ids, attention_mask).squeeze()
            loss = criterion(squeezed, labels)
            loss.backward()
            optimizer.step()
    except KeyError as e:
        print(f'Train missed key: {e}')

    # Validation
    model_to_train.eval()
    total_val_loss = 0
    with torch.no_grad():
        try:
            for batch in val_dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                squeezed = model_to_train(input_ids, attention_mask).squeeze()
                val_loss = criterion(squeezed, labels)
                total_val_loss += val_loss.item()
        except KeyError as e:
            print(f'Validate missed key: {e}')

    avg_val_loss = total_val_loss / len(val_dataloader)
    print(f"Training cycle {epoch + 1}, Validation Loss: {avg_val_loss:.4f}")

# Save the trained model
torch.save(model_to_train.state_dict(), "text_classifier_model.pth")
