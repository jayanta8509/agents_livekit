import asyncio
import logging
import os
import sys
from typing import Optional
import multiprocessing

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
    metrics,
)
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import openai, silero

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voice-assistant")

# Supported voices
# SUPPORTED_VOICES = ['onyx', 'nova', 'alloy', 'echo']

SUPPORTED_VOICES = ['alloy', 'ash', 'coral', 'echo', 'fable', 'onyx', 'nova', 'sage', 'shimmer']

class VoiceRequest(BaseModel):
    voice_name: str

def run_voice_assistant(voice_name: str):
    """
    Function to run the voice assistant in a separate process
    
    Args:
        voice_name (str): Name of the voice to use for text-to-speech
    """
    # Set up environment for the voice assistant
    os.environ['VOICE_NAME'] = voice_name.lower()

    # Directly use the entrypoint function
    def _run_app():
        cli.run_app(
            WorkerOptions(
                entrypoint_fnc=entrypoint,
                prewarm_fnc=prewarm,
            ),
        )

    # Run the app
    _run_app()

def prewarm(proc: JobProcess):
    """Prewarm the Voice Activity Detection model."""
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    # Retrieve voice name from environment variable
    voice_name = os.getenv('VOICE_NAME', 'onyx')
    
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=(
            "You are a voice assistant created by LiveKit. Your interface with users will be voice. "
            "You should use short responses."
        ),
    )

    logger.info(f"Connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    participant = await ctx.wait_for_participant()
    logger.info(f"Starting voice assistant with voice {voice_name}")

    agent = VoicePipelineAgent(
        vad=ctx.proc.userdata["vad"],
        stt=openai.STT(),
        llm=openai.LLM(),
        tts=openai.TTS(voice=voice_name),
        chat_ctx=initial_ctx,
    )

    agent.start(ctx.room, participant)
    usage_collector = metrics.UsageCollector()

    @agent.on("metrics_collected")
    def _on_metrics_collected(mtrcs: metrics.AgentMetrics):
        metrics.log_metrics(mtrcs)
        usage_collector.collect(mtrcs)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: ${summary}")

    ctx.add_shutdown_callback(log_usage)

    chat = rtc.ChatManager(ctx.room)

    async def answer_from_text(txt: str):
        chat_ctx = agent.chat_ctx.copy()
        chat_ctx.append(role="user", text=txt)
        stream = agent.llm.chat(chat_ctx=chat_ctx)
        await agent.say(stream)

    @chat.on("message_received")
    def on_chat_received(msg: rtc.ChatMessage):
        if msg.message:
            asyncio.create_task(answer_from_text(msg.message))

    await agent.say("Hey, how can I help you today?", allow_interruptions=True)

# Create FastAPI app
app = FastAPI()

@app.post("/start-voice-assistant")
async def start_assistant(request: VoiceRequest):
    """
    Endpoint to start the voice assistant with a specified voice
    
    Args:
        request (VoiceRequest): Request containing the voice name
    
    Returns:
        dict: Confirmation message
    
    Raises:
        HTTPException: If an unsupported voice is provided
    """
    # Validate voice name
    if request.voice_name.lower() not in SUPPORTED_VOICES:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported voice. Supported voices are: {', '.join(SUPPORTED_VOICES)}"
        )

    # Use multiprocessing to run the voice assistant
    process = multiprocessing.Process(
        target=run_voice_assistant, 
        args=(request.voice_name,)
    )
    process.start()

    return {"message": f"Voice assistant started with {request.voice_name} voice"}

@app.get("/supported-voices")
async def get_supported_voices():
    """
    Endpoint to retrieve list of supported voices
    
    Returns:
        dict: List of supported voices
    """
    return {"supported_voices": SUPPORTED_VOICES}

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()

    # Ensure multiprocessing works correctly on Windows
    multiprocessing.freeze_support()

    # Run the FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=8000)