import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import RobertaModel


class TextClassifier(nn.Module):
    def __init__(self):
        super(TextClassifier, self).__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.fc = nn.Linear(self.roberta.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids, attention_mask)
        return self.fc(outputs.pooler_output)


# Define a custom PyTorch dataset
class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.float32),
        }
