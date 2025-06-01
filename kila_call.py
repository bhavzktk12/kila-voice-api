# kila_call.py
# Production FastAPI app for handling voice calls via Twilio + OpenAI + ElevenLabs
# Uses same Pinecone memory as Instagram DMs for consistent conversation

import os
import json
import asyncio
import requests
import base64
import tempfile
import uuid
import glob
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import Response, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from twilio.twiml.voice_response import VoiceResponse, Gather, Play
from twilio.rest import Client
import uvicorn
from openai import OpenAI
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings

# Load environment variables from .env file
load_dotenv()

# ========================
# CONFIGURATION
# ========================

# Load environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY") 
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "agently-memory")

# Production configuration
BASE_URL = os.getenv("BASE_URL", "http://localhost:8001")  # Will be set to Render URL in production
PORT = int(os.getenv("PORT", 8001))  # Render assigns this automatically

# ElevenLabs Configuration
ELEVENLABS_VOICE_ID = "XrExE9yKIg1WjnnlVkGX"  # Your chosen voice
ELEVENLABS_MODEL = "eleven_turbo_v2_5"  # Fastest model for low latency

# Validate required environment variables
required_vars = [OPENAI_API_KEY, TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, PINECONE_API_KEY, ELEVENLABS_API_KEY]
if not all(required_vars):
    raise RuntimeError("Missing required environment variables")

# Initialize clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Initialize Pinecone (v5 compatible)
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    pinecone_index = pc.Index(PINECONE_INDEX_NAME)
    print(f"[INIT] Pinecone v5 initialized successfully with index: {PINECONE_INDEX_NAME}")
except Exception as e:
    print(f"[ERROR] Failed to initialize Pinecone: {str(e)}")
    raise

EMBEDDINGS = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

# Voice personality prompt (60% professional, 30% warm, 10% witty)
VOICE_SYSTEM_PROMPT = """
You are KILA, a world-class personal assistant with a sharp, professional voice. Your personality is:

60% PROFESSIONAL & BUSINESS-FOCUSED:
- Direct, efficient, and solution-oriented
- Use business terminology appropriately
- Always maintain composure and authority
- Focus on actionable outcomes

30% WARM & PERSONABLE:
- Genuinely care about the caller's needs
- Use their name when appropriate
- Show empathy and understanding
- Be approachable and friendly

10% WITTY & SHARP:
- Occasional subtle humor when appropriate
- Clever observations or insights
- Slight sarcasm only when it enhances the interaction
- Quick, intelligent responses

VOICE GUIDELINES:
- Keep responses very concise (1-2 sentences max)
- Speak with confidence and clarity
- Use natural speech patterns
- Avoid filler words like "um" or "uh"
- Be conversational but never casual
- End with a clear pause for the caller to respond

REMEMBER: You have access to the caller's conversation history. Reference previous conversations naturally when relevant.
"""

# ========================
# FASTAPI APP SETUP
# ========================

app = FastAPI(title="KILA Voice Call API - Production")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create audio directory and mount it
os.makedirs("audio_files", exist_ok=True)
app.mount("/audio", StaticFiles(directory="audio_files"), name="audio")

# ========================
# ELEVENLABS FUNCTIONS
# ========================

async def generate_elevenlabs_audio(text: str) -> str:
    """Generate audio using ElevenLabs and return the public URL."""
    try:
        print(f"[ELEVENLABS] Generating audio for: {text}")
        
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
        
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": ELEVENLABS_API_KEY
        }
        
        data = {
            "text": text,
            "model_id": ELEVENLABS_MODEL,
            "voice_settings": {
                "stability": 0.6,           # Slightly more stable for phone calls
                "similarity_boost": 0.8,    # Higher similarity to chosen voice
                "style": 0.3,               # Moderate style
                "use_speaker_boost": True   # Better for phone audio
            }
        }
        
        response = requests.post(url, json=data, headers=headers)
        
        if response.status_code == 200:
            # Create unique filename
            filename = f"{uuid.uuid4()}.mp3"
            filepath = f"audio_files/{filename}"
            
            # Save audio file
            with open(filepath, "wb") as f:
                f.write(response.content)
            
            # Return the public URL using BASE_URL (production URL)
            audio_url = f"{BASE_URL}/audio/{filename}"
            print(f"[ELEVENLABS] Audio saved to: {audio_url}")
            return audio_url
            
        else:
            print(f"[ELEVENLABS] Error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"[ELEVENLABS] Error generating audio: {str(e)}")
        return None

def cleanup_old_audio_files():
    """Clean up old audio files to save space."""
    try:
        audio_files = glob.glob("audio_files/*.mp3")
        current_time = time.time()
        
        for file_path in audio_files:
            file_age = current_time - os.path.getctime(file_path)
            if file_age > 3600:  # Delete files older than 1 hour
                os.remove(file_path)
                print(f"[CLEANUP] Removed old audio file: {file_path}")
    except Exception as e:
        print(f"[CLEANUP] Error cleaning up files: {str(e)}")

# ========================
# MEMORY FUNCTIONS (Pinecone v5 Compatible)
# ========================

def fetch_memory(phone_number: str) -> str:
    """Fetch conversation memory for a phone number from Pinecone v5."""
    try:
        print(f"[MEMORY] Fetching memory for phone: {phone_number}")
        resp = pinecone_index.fetch(ids=[phone_number], namespace="kila_voice")
        
        # Pinecone v5 compatible response handling
        if hasattr(resp, 'vectors') and resp.vectors:
            vectors = resp.vectors
            if phone_number in vectors:
                vector_data = vectors[phone_number]
                if hasattr(vector_data, 'metadata') and vector_data.metadata:
                    summary = vector_data.metadata.get('summary', '')
                    print(f"[MEMORY] Found memory for {phone_number}")
                    return summary
        
        print(f"[MEMORY] No memory found for {phone_number}")
        return ""
    except Exception as e:
        print(f"[ERROR] Memory fetch error: {str(e)}")
        return ""

def store_memory(phone_number: str, summary: str, interaction_type: str = "voice_call"):
    """Store conversation memory for a phone number in Pinecone v5."""
    try:
        print(f"[MEMORY] Storing memory for phone: {phone_number}")
        
        vector = EMBEDDINGS.embed_query(summary)
        
        metadata = {
            "summary": summary,
            "lastInteraction": datetime.now().isoformat(),
            "interaction_type": interaction_type,
            "phone_number": phone_number
        }
        
        # Pinecone v5 compatible upsert
        pinecone_index.upsert(
            vectors=[{
                "id": phone_number,
                "values": vector,
                "metadata": metadata
            }],
            namespace="kila_voice"
        )
        print(f"[MEMORY] Successfully stored memory for {phone_number}")
    except Exception as e:
        print(f"[ERROR] Memory storage error: {str(e)}")

# ========================
# VOICE CALL ENDPOINTS
# ========================

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "alive", 
        "service": "KILA Voice Call API - Production v5",
        "base_url": BASE_URL,
        "environment": "production" if "render.com" in BASE_URL else "development",
        "pinecone_version": "v5"
    }

@app.get("/health")
async def health_check():
    """Detailed health check for monitoring."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "openai": "connected" if OPENAI_API_KEY else "missing_key",
            "elevenlabs": "connected" if ELEVENLABS_API_KEY else "missing_key",
            "twilio": "connected" if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN else "missing_keys",
            "pinecone": "connected" if PINECONE_API_KEY else "missing_key"
        },
        "pinecone_version": "v5"
    }

@app.post("/call/incoming")
async def handle_incoming_call(request: Request):
    """Handle incoming phone calls from Twilio with ElevenLabs voice."""
    form_data = await request.form()
    caller_number = form_data.get("From", "unknown")
    
    print(f"[CALL] Incoming call from: {caller_number}")
    
    # Clean up old audio files
    cleanup_old_audio_files()
    
    # Create TwiML response
    response = VoiceResponse()
    
    # Generate greeting with ElevenLabs
    greeting = "Hello! This is KILA, your personal assistant. How can I help you today?"
    
    # Generate ElevenLabs audio for greeting
    audio_url = await generate_elevenlabs_audio(greeting)
    
    if audio_url:
        # Play ElevenLabs audio
        response.play(audio_url)
    else:
        # Fallback to Twilio voice
        response.say(greeting, voice="Polly.Joanna", language="en-US")
    
    # Use gather for real-time interaction
    gather = Gather(
        input='speech',
        timeout=4,
        speech_timeout='auto',
        action='/call/process_speech',
        method='POST'
    )
    
    response.append(gather)
    
    # If no speech detected, give a fallback
    response.say("I didn't hear anything. Please try calling again.", voice="Polly.Joanna")
    
    return Response(content=str(response), media_type="application/xml")

@app.post("/call/process_speech")
async def process_speech(request: Request):
    """Process speech input and continue conversation with ElevenLabs voice."""
    form_data = await request.form()
    caller_number = form_data.get("From", "unknown")
    speech_result = form_data.get("SpeechResult", "")
    
    print(f"[SPEECH] From {caller_number}: {speech_result}")
    
    response = VoiceResponse()
    
    if not speech_result:
        # Generate "didn't catch that" with ElevenLabs
        clarification_text = "I didn't catch that. Could you please repeat?"
        audio_url = await generate_elevenlabs_audio(clarification_text)
        
        if audio_url:
            response.play(audio_url)
        else:
            response.say(clarification_text, voice="Polly.Joanna")
        
        gather = Gather(
            input='speech',
            timeout=4,
            speech_timeout='auto',
            action='/call/process_speech',
            method='POST'
        )
        response.append(gather)
        return Response(content=str(response), media_type="application/xml")
    
    # Fetch memory for this phone number
    memory = fetch_memory(caller_number)
    
    # Build conversation context
    chat_messages = [{"role": "system", "content": VOICE_SYSTEM_PROMPT}]
    
    if memory:
        chat_messages.append({
            "role": "system", 
            "content": f"Previous conversation context: {memory}"
        })
    
    chat_messages.append({"role": "user", "content": speech_result})
    
    # Get AI response
    try:
        ai_response = openai_client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=chat_messages,
            temperature=0.6,
            max_tokens=50  # Short for phone conversations
        )
        reply = ai_response.choices[0].message.content.strip()
        print(f"[AI] Response: {reply}")
    except Exception as e:
        print(f"[ERROR] AI response error: {str(e)}")
        reply = "I'm sorry, I'm having trouble processing that right now."
    
    # Update memory
    try:
        if memory:
            updated_memory = f"{memory}\nCaller: {speech_result}\nKILA: {reply}"
        else:
            updated_memory = f"Phone: {caller_number}\nCaller: {speech_result}\nKILA: {reply}"
        
        store_memory(caller_number, updated_memory)
    except Exception as e:
        print(f"[ERROR] Memory update error: {str(e)}")
    
    # Generate response with ElevenLabs
    audio_url = await generate_elevenlabs_audio(reply)
    
    if audio_url:
        # Play ElevenLabs audio
        response.play(audio_url)
    else:
        # Fallback to Twilio voice
        response.say(reply, voice="Polly.Joanna", language="en-US")
    
    # Continue the conversation
    gather = Gather(
        input='speech',
        timeout=6,
        speech_timeout='auto',
        action='/call/process_speech',
        method='POST'
    )
    response.append(gather)
    
    # If they don't respond, end gracefully with ElevenLabs
    goodbye_text = "Thank you for calling. Have a great day!"
    goodbye_url = await generate_elevenlabs_audio(goodbye_text)
    
    if goodbye_url:
        response.play(goodbye_url)
    else:
        response.say(goodbye_text, voice="Polly.Joanna")
    
    return Response(content=str(response), media_type="application/xml")

@app.post("/call/hangup")
async def handle_hangup(request: Request):
    """Handle when the caller hangs up."""
    form_data = await request.form()
    caller_number = form_data.get("From", "unknown")
    
    print(f"[CALL] Call ended with {caller_number}")
    return {"status": "call_ended"}

# ========================
# MAIN EXECUTION
# ========================

if __name__ == "__main__":
    print(f"[STARTUP] Starting KILA Voice Call API on port {PORT}...")
    uvicorn.run("kila_call:app", host="0.0.0.0", port=PORT, reload=False)