import torch
import torch.nn as nn
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertModel
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

# Dataset
data = {
    "input_text": [...],  # Same as before
    "label": [...]
}
df = pd.DataFrame(data)

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
tokenized = tokenizer(list(df['input_text']), padding=True, truncation=True, max_length=64, return_tensors="pt")

label2id = {label: idx for idx, label in enumerate(df['label'].unique())}
labels = torch.tensor(df['label'].map(label2id).values)

class LifeAdvisorModel(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output.last_hidden_state[:, 0]
        x = self.dropout(hidden_state)
        return self.classifier(x)

class AdviceDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }

dataset = AdviceDataset(tokenized['input_ids'], tokenized['attention_mask'], labels)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LifeAdvisorModel(num_labels=len(label2id)).to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)
criterion = nn.CrossEntropyLoss()

model.train()
for epoch in range(3):
    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        label = batch['labels'].to(device)

        optimizer.zero_grad()
        output = model(input_ids, attention_mask)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

# Save model and mappings
torch.save({
    'model_state_dict': model.state_dict(),
    'label2id': label2id
}, 'life_advice_model.pth')

print("Model trained and saved as life_advice_model.pth ✅")
