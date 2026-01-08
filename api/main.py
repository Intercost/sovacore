import os
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

app = FastAPI(
    title="Sova Chatbot API",
    description="SovaCore's official AI assistant. Headless API for web integration.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace "*" with your actual vercel domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("ERROR: GEMINI_API_KEY environment variable not set. Please configure it.")
    # In a production setup, you might want to exit or handle this more gracefully.
    # For Vercel, this will likely lead to a deployment error, which is desired.
    exit(1)

genai.configure(api_key=GEMINI_API_KEY)

SOVACORE_KNOWLEDGE = """
You are SOVA, the official AI assistant for SovaCore - a cutting-edge AI automation company.

## About SovaCore
SovaCore deploys autonomous AI agents as a service. We help businesses achieve superhuman efficiency through AI automation. Our mission is to transform how businesses operate by deploying intelligent, autonomous AI agents that work 24/7.

## Services
1. **AI Readiness Audit** - We assess your business processes and identify automation opportunities
2. **Custom AI Agent Development** - We build tailored AI solutions for your specific needs
3. **AI Agent Marketplace** - Pre-built AI agents ready for deployment
4. **Discovery Calls** - Free consultations to understand your needs

## Pricing Tiers
- **Starter**: For small businesses beginning their AI journey
- **Professional**: For growing companies with moderate automation needs
- **Enterprise**: Full-scale AI transformation with dedicated support

## Key Features
- 24/7 Autonomous Operation
- Natural Language Understanding
- Multi-platform Integration
- Real-time Analytics & Insights
- Custom Training & Fine-tuning
- Enterprise-grade Security

## Case Studies
We've helped businesses across industries achieve:
- 80% reduction in manual tasks
- 24/7 customer support coverage
- 10x faster data processing
- Significant cost savings

## Ambassador Program
Join our community and earn rewards by referring businesses to SovaCore.

## Contact
Website: SovaCore (the website the user is on)
Discovery calls available for free consultations.

Remember: You ARE SovaCore's AI - speak confidently about "our" services, "we" offer, etc. Be helpful, knowledgeable, and represent the brand with a futuristic, professional tone.
"""

# Configure the Generative Model
generation_config = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 60,
    "max_output_tokens": 1024,
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

# Initialize the Gemini model with system instruction globally
# This ensures the model is loaded once when the application starts.
model = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    generation_config=generation_config,
    safety_settings=safety_settings,
    system_instruction=SOVACORE_KNOWLEDGE
)

# Pydantic model for conversation history parts
class HistoryPart(BaseModel):
    text: str

# Pydantic model for a single history entry
class HistoryEntry(BaseModel):
    role: str
    parts: List[HistoryPart]

# Pydantic model for incoming chat requests
class ChatRequest(BaseModel):
    message: str
    history: List[HistoryEntry] = [] # List of previous user/model exchanges

@app.post("/chat")
async def chat_with_sova(request: ChatRequest):
    """
    Handles incoming chat messages and returns Sova's response along with updated conversation history.
    """
    try:
        # Convert Pydantic history objects back to the format expected by genai.start_chat
        genai_history = []
        for entry in request.history:
            parts = []
            for p_part in entry.parts:
                parts.append({"text": p_part.text}) # Assuming only text parts for now
            genai_history.append({"role": entry.role, "parts": parts})
        
        # Start a new chat session with the provided history. The system_instruction is persistent.
        chat_session = model.start_chat(history=genai_history)
        
        # Send the current user message to the chat session
        response = chat_session.send_message(request.message)
        
        # Check if the response was blocked by safety settings
        if response.candidates and response.candidates[0].finish_reason == 4: # SAFETY_SETTING_BLOCK
            return {
                "response": "I'm sorry, I cannot respond to that query due to safety concerns. Please try rephrasing your question.",
                "history": request.history # Return original history if response was blocked
            }

        # Convert the full chat_session.history (which includes the latest turn)
        # back into the Pydantic model format to send to the client.
        updated_history: List[HistoryEntry] = []
        for content in chat_session.history:
            parts_list = [HistoryPart(text=part.text) for part in content.parts if hasattr(part, 'text')]
            updated_history.append(HistoryEntry(role=content.role, parts=parts_list))

        return {
            "response": response.text,
            "history": updated_history
        }
    except Exception as e:
        print(f"An error occurred during chat processing: {e}") # Log error for debugging
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

@app.get("/", summary="Health Check")
async def root():
    """
    A simple health check endpoint to confirm the API is running.
    """
    return {
        "message": "Sova Chatbot API is running. Use the /chat endpoint to interact.",
        "version": app.version,
        "model": model.model_name
    }
