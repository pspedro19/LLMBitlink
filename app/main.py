from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from torch.nn import DataParallel
import mlflow.pytorch
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
import uvicorn
import os

# Load environment variables using a relative path
# Assuming your .env file is two levels up from the main.py script
env_path = Path(__file__).resolve().parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# Environment Variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("OpenAI API key must be set in the environment variables.")

HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
if not HF_TOKEN:
    raise EnvironmentError("Hugging Face token must be set in the environment variables.")

# FastAPI app
app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent

class SimpleLLM:
    def __init__(self, model=None, tokenizer=None, client=None):
        self.model = model
        self.tokenizer = tokenizer
        self.client = client

    async def generate_text(self, input_text: str) -> str:
        if self.client:  # Using OpenAI API
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": input_text}]
            )
            return response.choices[0].message.content
        elif self.tokenizer and self.model:  # Use local models if OpenAI is not available
            input_ids = self.tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=1024)
            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
            output = self.model.generate(
                input_ids,
                max_new_tokens=50,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                pad_token_id=self.tokenizer.eos_token_id,
                num_beams=5,
                early_stopping=True
            )
            return self.tokenizer.decode(output[0], skip_special_tokens=True)

class ChatInput(BaseModel):
    user_input: str

class ChatOutput(BaseModel):
    response: str

def load_from_openai():
    client = OpenAI(api_key=OPENAI_API_KEY)
    return SimpleLLM(client=client)

def select_model_source():
    try:
        print("Attempting to load model from OpenAI...")
        return load_from_openai()
    except Exception as e:
        print(f"Failed to load model from OpenAI: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to load model from any source")

@app.on_event("startup")
async def startup_event():
    global llm
    llm = select_model_source()

@app.post("/chat/", response_model=ChatOutput)
async def chat_with_agent(chat_input: ChatInput):
    response = await llm.generate_text(chat_input.user_input)
    return ChatOutput(response=response)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("FASTAPI_PORT", 8800)))
