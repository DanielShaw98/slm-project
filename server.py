from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from transformers import pipeline
import torch
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
app = FastAPI()
token_auth_scheme = HTTPBearer()

# Retrieve the Hugging Face token from environment variable
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Initialize the pipeline
model_id = "DanielShaw98/phi-3.5-law"
model_pipeline = pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto"  # Requires accelerate to work properly
)

class Message(BaseModel):
    role: str
    content: str

class RequestBody(BaseModel):
    messages: list[Message]
    max_new_tokens: int

# Function to verify token
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(token_auth_scheme)):
    token = credentials.credentials
    if token != HUGGINGFACE_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing token")
    return token

@app.post("/generate")
async def generate_text(request_body: RequestBody, token: str = Depends(verify_token)):
    try:
        outputs = model_pipeline(
            [{"role": msg.role, "content": msg.content} for msg in request_body.messages],
            max_new_tokens=request_body.max_new_tokens,
        )
        return {"generated_text": outputs[0]["generated_text"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
