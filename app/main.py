# main.py

import openai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv
import os

# Specify the path to the .env file and load environment variables
env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path=env_path)

# Initialize the FastAPI app
app = FastAPI()

# Set up OpenAI client configuration with environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")
engine = 'gpt-3.5-turbo'  # Adjust according to your subscription and needs

# Define the Pydantic model for chat message input validation
class ChatMessage(BaseModel):
    user_input: str

# Function to generate text completions using the OpenAI API
def get_completion(prompt, engine="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=engine,
        messages=messages,
        temperature=0.5  # Controls the randomness of the model's output
    )
    return response.choices[0].message["content"]

# Define a POST endpoint for the chat interaction
@app.post("/chat/")
async def chat_with_agent(chat_message: ChatMessage):
    try:
        result = get_completion(chat_message.user_input, engine)
        return {"response": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Define a GET endpoint to verify server operation
@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI OpenAI Integration!"}

# If the script is executed as the main script, run the server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8800, reload=True)
