import asyncio
import sys
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn import DataParallel
from memory import EnhancedVectorMemory  # Ensure memory.py is accessible from this script, adjust the import path as needed.

app = FastAPI()

class SimpleLLM:
    def __init__(self, model_path: str, tokenizer_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if torch.cuda.is_available():
            self.model = AutoModelForCausalLM.from_pretrained(model_path).cuda()
            self.model = DataParallel(self.model)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_path)

    async def generate_text(self, input_text: str, memory_context: Optional[str] = None) -> str:
        input_text = f"{memory_context} {input_text}" if memory_context else input_text
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=1024)
        input_ids = input_ids.cuda() if torch.cuda.is_available() else input_ids

        actual_model = self.model.module if isinstance(self.model, DataParallel) else self.model
        output = actual_model.generate(
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

# Configuration paths
tokenizer_path = "/DnlLLM/src/DnlModel/DnlModel"
model_path = "/DnlLLM/src/DnlModel/checkpoint-225"
memory_path = "/DnlLLM/data/memory.json"

# Initialization of components
memory = EnhancedVectorMemory(memory_path)
llm = SimpleLLM(model_path, tokenizer_path)

@app.post("/chat/", response_model=ChatOutput)
async def chat_with_agent(chat_input: ChatInput):
    try:
        memory_response = memory.get_closest_memory(chat_input.user_input)
        response = await llm.generate_text(chat_input.user_input, memory_response)
        memory.add_to_memory(chat_input.user_input, response)  # Verify thread-safety if used in async context.
        return ChatOutput(response=response)
    except Exception as e:
        # It's usually a good practice to log the exception details here.
        raise HTTPException(status_code=500, detail=str(e))

async def run_sales_agent():
    print("Welcome! You're chatting with DNL Agent. How may I assist you today?")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("DNL Agent: It was a pleasure assisting you. Goodbye!")
            break
        try:
            memory_response = memory.get_closest_memory(user_input)
            response = await llm.generate_text(user_input, memory_response)
            memory.add_to_memory(user_input, response)  # Assuming add_to_memory is not async
            print("DNL Agent:", response)
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    import uvicorn
    if "serve" in sys.argv:
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    else:
        asyncio.run(run_sales_agent())
