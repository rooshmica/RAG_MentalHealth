from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pathlib import Path
import os

# Load environment variables
print(os.getcwd())
load_dotenv(Path(os.getcwd()+"/.env"))

origins = ["http://localhost:3000"]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the fine-tuned emotion detection model
MODEL_PATH = "./emotion_model"
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)

# Define request model
class TextRequest(BaseModel):
    text: str

# Function to predict emotion
def detect_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.sigmoid(outputs.logits).squeeze().tolist()
    emotion_index = probs.index(max(probs))  # Get the most probable emotion index
    return emotion_index

@app.post("/analyze_emotion")
async def analyze_emotion(request: TextRequest):
    """Detects emotions in user text using the fine-tuned model."""
    emotion = detect_emotion(request.text)
    return {"emotion": emotion}

@app.post("/generate_response")
async def generate_response(request: TextRequest):
    """Generates an AI response based on the detected emotion."""
    emotion_result = await analyze_emotion(request)
    emotion = emotion_result["emotion"]

    # Generate a response based on the detected emotion
    response_text = f"The detected emotion is {emotion}. How can I assist you further?"
    
    return {"response": response_text}

@app.get("/status")
async def status():
    return {"status": "Server is running"}