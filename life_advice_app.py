import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from transformers import DistilBertTokenizer
from transformers import DistilBertModel
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import random

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

data = {
    "input_text": [
        # Career
        "I'm considering a career change",
        "How do I get started in data science?",
        "What should I do if I feel stuck in my job?",
        "How do I prepare for a leadership role?",

        # Mental Health
        "I feel stressed about my responsibilities",
        "I need to find a way to relax after work",
        "How can I stop overthinking?",
        "I'm feeling burned out, what should I do?",

        # Relationships
        "I'm struggling to communicate with my partner",
        "What should I do when we disagree?",
        "How do I maintain strong friendships?",
        "How can I set healthy boundaries with others?",

        # Productivity
        "I'm procrastinating, how do I get started?",
        "What’s the best way to stay focused at work?",
        "How can I organize my tasks better?",
        "I need help creating a productive morning routine?",

        # Self Growth
        "How do I stay motivated when I feel stuck?",
        "What can I do to grow personally every day?",
        "How do I improve my confidence?",
        "I need to stop comparing myself to others, how do I do that?",

        # Fitness
        "How do I make fitness a habit?",
        "What are some simple ways to stay active during the day?",
        "Should I exercise every day or take breaks?",
        "How can I make the most out of my workout?",

        # Finance
        "What are some good budgeting tips?",
        "How do I start saving money for retirement?",
        "Should I invest in stocks or real estate?",
        "How can I avoid living paycheck to paycheck?",

        # Education
        "How do I learn effectively for exams?",
        "What are some tips for staying focused during lectures?",
        "Should I study with a group or alone?",
        "How do I retain information better for long-term learning?",

        # Time Management
        "How do I manage my time effectively?",
        "I’m overwhelmed with tasks, how can I prioritize better?",
        "What are the best tools for tracking my time?",
        "How can I avoid distractions during work?",

        # Motivation
        "How do I stay motivated when I'm feeling lazy?",
        "What can I do when I lose motivation?",
        "How do I set achievable goals?",
        "How do I stay positive during tough times?"
    ],
    "label": [
        # Career
        "career", "career", "career", "career",

        # Mental Health
        "mental_health", "mental_health", "mental_health", "mental_health",

        # Relationships
        "relationships", "relationships", "relationships", "relationships",

        # Productivity
        "productivity", "productivity", "productivity", "productivity",

        # Self Growth
        "self_growth", "self_growth", "self_growth", "self_growth",

        # Fitness
        "fitness", "fitness", "fitness", "fitness",

        # Finance
        "finance", "finance", "finance", "finance",

        # Education
        "education", "education", "education", "education",

        # Time Management
        "time_management", "time_management", "time_management", "time_management",

        # Motivation
        "motivation", "motivation", "motivation", "motivation"
    ]
}



df = pd.DataFrame(data)
df.to_csv("life_advice.csv", index=False)


tokenized = tokenizer(
    list(df['input_text']),
    padding=True,
    truncation=True,
    max_length=64,
    return_tensors="pt"
)
label2id = {label: idx for idx, label in enumerate(df['label'].unique())}
id2label = {v: k for k, v in label2id.items()}
labels = df['label'].map(label2id).values


class LifeAdvisorModel(nn.Module):
    def __init__(self, num_labels):
        super(LifeAdvisorModel, self).__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = outputs.last_hidden_state[:, 0]  # [CLS] token
        pooled_output = self.dropout(hidden_state)
        return self.classifier(pooled_output)


input_ids = tokenized['input_ids']
attention_mask = tokenized['attention_mask']
labels_tensor = torch.tensor(labels)

print("Input shape:", input_ids.shape)
print("Attention mask shape:", attention_mask.shape)
print("Labels shape:", labels_tensor.shape)

num_labels = len(label2id)
model = LifeAdvisorModel(num_labels)



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

dataset = AdviceDataset(input_ids, attention_mask, labels_tensor)
loader = DataLoader(dataset, batch_size=2, shuffle=True)



optimizer = AdamW(model.parameters(), lr=5e-5)
criterion = nn.CrossEntropyLoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

model.train()
epochs = 5

for epoch in range(epochs):
    total_loss = 0

    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss:.4f}")
advice_texts = {
    "career": [
        "It’s okay to change direction — your path isn’t set in stone.",
        "Explore side projects that excite you. They often open doors.",
        "Talk to people in fields you're curious about. Insight = clarity.",
        "Don’t chase titles, chase learning — growth compounds.",
        "Everyone starts somewhere. Progress beats perfection."
    ],
    "mental_health": [
        "Take it one day at a time — and remember to be kind to yourself.",
        "Journaling can help you understand what’s really going on inside.",
        "You're not alone — reaching out is a sign of strength, not weakness.",
        "Mental health days are valid — rest is productive too.",
        "Even small steps toward healing matter. Keep going."
    ],
    "relationships": [
        "Honest conversations build stronger connections.",
        "It’s okay to disagree — respect matters more than winning.",
        "If someone matters, make the effort to understand them.",
        "Trust grows with consistency and care, not just words.",
        "Set boundaries that protect your peace — it's healthy."
    ],
    "productivity": [
        "Try the 5-minute rule: start small, keep the momentum.",
        "Energy > time. Work when you're sharp, rest when you’re drained.",
        "Clear your space, clear your mind. Environment matters.",
        "Batch similar tasks to stay in flow longer.",
        "Make progress visible. Checking things off = dopamine boost."
    ],
    "self_growth": [
        "You don’t have to have it all figured out — just keep learning.",
        "Reflect often, act intentionally. Growth follows awareness.",
        "Comparison steals joy — measure your path against your past.",
        "Get comfortable with discomfort. That’s where the magic happens.",
        "Mistakes are feedback, not failure."
    ],
    "fitness": [
        "Consistency beats intensity. Show up — even for 15 minutes.",
        "Fuel your body like it’s the only one you get — because it is.",
        "Find a movement you enjoy. Fun = sustainable.",
        "Rest days are part of the plan — not a break from it.",
        "Progress looks different for everyone. Honor yours."
    ],
    "finance": [
        "Start small: even saving $5 builds the habit.",
        "Track your spending to uncover hidden patterns.",
        "Avoid lifestyle creep — invest in your future self.",
        "Build an emergency fund before chasing big gains.",
        "Money is a tool, not a scoreboard. Use it wisely."
    ],
    "education": [
        "Study smarter, not longer — active recall beats re-reading.",
        "Teach someone else — it’s the best way to solidify your learning.",
        "Don’t fear asking questions — curiosity is a strength.",
        "Spacing out study sessions beats cramming every time.",
        "Make it relevant. Tie what you learn to your real life."
    ],
    "time_management": [
        "Time-blocking helps protect your focus from distractions.",
        "Prioritize the important over the urgent.",
        "You don’t need more time — just better priorities.",
        "Leave buffer time between tasks. Life happens.",
        "Track your time for a day — awareness changes everything."
    ],
    "motivation": [
        "Motivation comes *after* action, not before.",
        "Visualize the ‘why’ behind your goals every day.",
        "Break it down. One small win can reignite your spark.",
        "You won’t always feel like it — show up anyway.",
        "You’re allowed to pause. Just don’t quit."
    ]
}

def predict_advice(text):
    model.eval()

    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="pt"
    )

    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        predicted_class = torch.argmax(outputs, dim=1).item()

    return id2label[predicted_class]


def get_advice(text):
    model.eval()

    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="pt"
    )

    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        predicted_class = torch.argmax(outputs, dim=1).item()

    label = id2label[predicted_class]
    return random.choice(advice_texts[label])






# Streamlit UI
st.title("🧠 Life Advice AI")
user_input = st.text_area("What's on your mind? Ask for some life advice:")

if st.button("Get Advice"):
    if user_input.strip() != "":
        label, advices = predict_advice(user_input)
        st.markdown(f"**Category:** `{label.replace('_', ' ').title()}`")
        st.markdown("**Advice for you:**")
        for adv in advices:
            st.markdown(f"- {adv}")
    else:
        st.warning("Please enter something to get advice.")
