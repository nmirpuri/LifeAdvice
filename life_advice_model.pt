import streamlit as st
import torch
from transformers import DistilBertTokenizer, DistilBertModel
import torch.nn as nn
import torch.nn.functional as F

# Load tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Define label mapping
label2id = {
    "career": 0,
    "mental_health": 1,
    "relationships": 2,
    "productivity": 3,
    "self_growth": 4,
    "fitness": 5,
    "finance": 6,
    "education": 7,
    "time_management": 8,
    "motivation": 9
}
id2label = {v: k for k, v in label2id.items()}

# Advice dictionary
advice_texts = {
    "career": [
        "It’s okay to change direction — your path isn’t set in stone.",
        "Everyone starts somewhere. Progress beats perfection."
    ],
    "mental_health": [
        "Take it one day at a time — and remember to be kind to yourself.",
        "Mental health days are valid — rest is productive too."
    ],
    "relationships": [
        "Honest conversations build stronger connections.",
        "Set boundaries that protect your peace — it's healthy."
    ],
    "productivity": [
        "Try the 5-minute rule: start small, keep the momentum.",
        "Make progress visible. Checking things off = dopamine boost."
    ],
    "self_growth": [
        "You don’t have to have it all figured out — just keep learning.",
        "Mistakes are feedback, not failure."
    ],
    "fitness": [
        "Consistency beats intensity. Show up — even for 15 minutes.",
        "Find a movement you enjoy. Fun = sustainable."
    ],
    "finance": [
        "Start small: even saving $5 builds the habit.",
        "Money is a tool, not a scoreboard. Use it wisely."
    ],
    "education": [
        "Study smarter, not longer — active recall beats re-reading.",
        "Spacing out study sessions beats cramming every time."
    ],
    "time_management": [
        "Time-blocking helps protect your focus from distractions.",
        "Track your time for a day — awareness changes everything."
    ],
    "motivation": [
        "Motivation comes *after* action, not before.",
        "You’re allowed to pause. Just don’t quit."
    ]
}

# Define model
class LifeAdvisorModel(nn.Module):
    def __init__(self, num_labels):
        super(LifeAdvisorModel, self).__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(hidden_state)
        return self.classifier(pooled_output)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LifeAdvisorModel(num_labels=len(label2id))
model.load_state_dict(torch.load("life_advice_model.pt", map_location=device))
model.to(device)
model.eval()

# Prediction function
def predict_advice(text):
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
    return label, advice_texts[label]

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
