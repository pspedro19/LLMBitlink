import openai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

# Set up OpenAI client configuration
openai.api_key = "c4129d4dad0247a691ad29bd38db1260"
openai.api_base = "https://openaiuniandesdevtest.openai.azure.com/"
engine = 'gpt-turbo-RF-NT'

app = FastAPI()

class ChatMessage(BaseModel):
    user_input: str

# Function to generate text completions using the OpenAI API
def get_completion(prompt, engine="gpt-turbo-RF-NT"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        engine=engine,
        messages=messages,
        temperature=0.5  # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]

@app.post("/chat/")
async def chat_with_agent(chat_message: ChatMessage):
    try:
        result = get_completion(chat_message.user_input, engine)
        return {"response": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI OpenAI Integration!"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8800, reload=True)
