import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertModel
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import random

# Load tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Define dataset
data = {
    "input_text": [
        # Career
        "I'm considering a career change", "How do I get started in data science?",
        "What should I do if I feel stuck in my job?", "How do I prepare for a leadership role?",
        # Mental Health
        "I feel stressed about my responsibilities", "I need to find a way to relax after work",
        "How can I stop overthinking?", "I'm feeling burned out, what should I do?",
        # Relationships
        "I'm struggling to communicate with my partner", "What should I do when we disagree?",
        "How do I maintain strong friendships?", "How can I set healthy boundaries with others?",
        # Productivity
        "I'm procrastinating, how do I get started?", "What’s the best way to stay focused at work?",
        "How can I organize my tasks better?", "I need help creating a productive morning routine?",
        # Self Growth
        "How do I stay motivated when I feel stuck?", "What can I do to grow personally every day?",
        "How do I improve my confidence?", "I need to stop comparing myself to others, how do I do that?",
        # Fitness
        "How do I make fitness a habit?", "What are some simple ways to stay active during the day?",
        "Should I exercise every day or take breaks?", "How can I make the most out of my workout?",
        # Finance
        "What are some good budgeting tips?", "How do I start saving money for retirement?",
        "Should I invest in stocks or real estate?", "How can I avoid living paycheck to paycheck?",
        # Education
        "How do I learn effectively for exams?", "What are some tips for staying focused during lectures?",
        "Should I study with a group or alone?", "How do I retain information better for long-term learning?",
        # Time Management
        "How do I manage my time effectively?", "I’m overwhelmed with tasks, how can I prioritize better?",
        "What are the best tools for tracking my time?", "How can I avoid distractions during work?",
        # Motivation
        "How do I stay motivated when I'm feeling lazy?", "What can I do when I lose motivation?",
        "How do I set achievable goals?", "How do I stay positive during tough times?"
    ],
    "label": [
        "career", "career", "career", "career",
        "mental_health", "mental_health", "mental_health", "mental_health",
        "relationships", "relationships", "relationships", "relationships",
        "productivity", "productivity", "productivity", "productivity",
        "self_growth", "self_growth", "self_growth", "self_growth",
        "fitness", "fitness", "fitness", "fitness",
        "finance", "finance", "finance", "finance",
        "education", "education", "education", "education",
        "time_management", "time_management", "time_management", "time_management",
        "motivation", "motivation", "motivation", "motivation"
    ]
}

df = pd.DataFrame(data)

# Tokenize and encode
tokenized = tokenizer(
    list(df['input_text']),
    padding=True,
    truncation=True,
    max_length=64,
    return_tensors="pt"
)

label2id = {label: idx for idx, label in enumerate(df['label'].unique())}
id2label = {v: k for k, v in label2id.items()}
labels = torch.tensor(df['label'].map(label2id).values)

# Model
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

# Dataset
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

# Train
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LifeAdvisorModel(num_labels=len(label2id)).to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)
criterion = nn.CrossEntropyLoss()

model.train()
for epoch in range(3):  # Keep training short for demo
    total_loss = 0
    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        label = batch['labels'].to(device)

        optimizer.zero_grad()
        output = model(input_ids, attention_mask)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} | Loss: {total_loss:.4f}")

# Advice tips
advice_texts = {
    "career": [
        "Explore side projects that excite you.", "Don’t chase titles, chase growth."
    ],
    "mental_health": [
        "Take it one day at a time.", "You're not alone — reaching out helps."
    ],
    "relationships": [
        "Honest conversations build trust.", "Respect matters more than being right."
    ],
    "productivity": [
        "Try the 5-minute rule.", "Clear your space, clear your mind."
    ],
    "self_growth": [
        "You don’t have to have it all figured out.", "Mistakes are feedback."
    ],
    "fitness": [
        "Consistency > intensity.", "Find a movement you enjoy."
    ],
    "finance": [
        "Track your spending.", "Start saving early — even small amounts."
    ],
    "education": [
        "Active recall beats rereading.", "Teach someone to learn better."
    ],
    "time_management": [
        "Time-block to protect focus.", "You need better priorities, not more time."
    ],
    "motivation": [
        "Action sparks motivation.", "You’re allowed to pause — not quit."
    ]
}

# Prediction function
def predict_advice(text):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=64)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        predicted_class_id = torch.argmax(logits, dim=1).item()

    predicted_label = id2label[predicted_class_id]
    return predicted_label, random.choice(advice_texts[predicted_label])

# Streamlit UI
st.title("🌱 Life Advice Assistant")
user_input = st.text_input("What's on your mind? (e.g. 'I'm stressed about work')")

if user_input:
    category, advice = predict_advice(user_input)
    st.markdown(f"**Category**: `{category}`")
    st.markdown(f"💡 **Advice**: {advice}")
