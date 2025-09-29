import modal
import base64
from typing import Dict

stub = modal.Stub("voice-synthesis-fast")

# GPU image with Coqui TTS
image = modal.Image.debian_slim().pip_install(
    "TTS==0.22.0",
    "torch==2.0.1",
    "numpy==1.24.3"
)

# Persistent storage for voice models
volume = modal.NetworkFileSystem.persisted("voice-models-storage")

@stub.function(
    image=image,
    gpu="A10G",
    network_file_systems={"/models": volume},
    timeout=60,
    container_idle_timeout=10  # Shutdown after 10s idle
)
def synthesize_speech(text: str, voice_id: str = "default") -> Dict:
    """Generate speech in 2-3 seconds using cloned voice"""
    from TTS.api import TTS
    import time
    import io
    import soundfile as sf
    
    start_time = time.time()
    
    # Load TTS with your cloned voice model
    # For testing, use default model first
    if voice_id == "default":
        tts = TTS("tts_models/en/ljspeech/tacotron2-DDC")
    else:
        # Load your custom XTTS model
        tts = TTS(model_path=f"/models/{voice_id}/model.pth",
                  config_path=f"/models/{voice_id}/config.json")
    
    # Generate audio
    wav = tts.tts(text)
    
    # Convert to base64 for API response
    buffer = io.BytesIO()
    sf.write(buffer, wav, 22050, format='WAV')
    audio_base64 = base64.b64encode(buffer.getvalue()).decode()
    
    generation_time = time.time() - start_time
    
    return {
        "audio_base64": audio_base64,
        "generation_time": generation_time,
        "voice_id": voice_id
    }

@stub.function(image=image, network_file_systems={"/models": volume})
def upload_voice_model(model_data: bytes, config_data: bytes, voice_id: str):
    """Upload a trained voice model to Modal storage"""
    import os
    
    model_dir = f"/models/{voice_id}"
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model files
    with open(f"{model_dir}/model.pth", "wb") as f:
        f.write(model_data)
    
    with open(f"{model_dir}/config.json", "wb") as f:
        f.write(config_data)
    
    return f"Model {voice_id} uploaded successfully"

@stub.local_entrypoint()
def test_synthesis():
    """Test function - run with: modal run modal_app.py"""
    result = synthesize_speech.remote(
        text="Hello! This is a test of the voice synthesis system.",
        voice_id="default"
    )
    print(f"Generated audio in {result['generation_time']:.2f} seconds")
    print(f"Audio size: {len(result['audio_base64'])} bytes (base64)")
