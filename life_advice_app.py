import streamlit as st
import torch
import torch.nn as nn
import random
from transformers import DistilBertTokenizer, DistilBertModel

# Load tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

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

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load('life_advice_model.pth', map_location=device)
label2id = checkpoint['label2id']
id2label = {v: k for k, v in label2id.items()}

model = LifeAdvisorModel(num_labels=len(label2id))
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# Advice bank
advice_texts = {
    "career": [...],
    "mental_health": [...],
    "relationships": [...],
    "productivity": [...],
    "self_growth": [...],
    "fitness": [...],
    "finance": [...],
    "education": [...],
    "time_management": [...],
    "motivation": [...]
}

# Predict
def predict_advice(text):
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
