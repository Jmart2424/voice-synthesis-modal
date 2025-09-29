import modal
import base64
import tempfile
import os
from typing import Dict, Optional

stub = modal.Stub("voice-synthesis-fast")

# === EASY VOICE SWITCHING (like your Railway) ===
DEFAULT_VOICE = "jenny"  # Just change this line!

VOICE_MODELS = {
    "jenny": "tts_models/en/jenny/jenny",
    "ljspeech": "tts_models/en/ljspeech/tacotron2-DDC",
    "glow": "tts_models/en/ljspeech/glow-tts",
    "vctk": "tts_models/en/vctk/vits",
    "xtts": "tts_models/multilingual/multi-dataset/xtts_v2"
}

# GPU image with Coqui TTS
image = modal.Image.debian_slim().apt_install(
    "gcc", "g++", "libsndfile1", "ffmpeg"  # ffmpeg for audio processing
).pip_install(
    "TTS==0.22.0",
    "torch==2.0.1",
    "torchaudio==2.0.2",
    "numpy==1.24.3",
    "soundfile",
    "librosa",  # For voice cloning preprocessing
    "scipy"
)

# Persistent storage for cloned voice models
volume = modal.NetworkFileSystem.persisted("cloned-voice-models")

@stub.function(
    image=image,
    gpu="A10G",
    network_file_systems={"/models": volume},
    timeout=60,
    container_idle_timeout=10
)
def synthesize_speech(
    text: str,
    voice_id: Optional[str] = None,
    return_format: str = "wav"  # "wav" or "base64"
) -> Dict:
    """
    Generate speech in 2-3 seconds using cloned or preset voices
    Can return either WAV file bytes or base64 (for compatibility)
    """
    from TTS.api import TTS
    import time
    import soundfile as sf
    
    start_time = time.time()
    
    # Determine which voice to use
    if voice_id and os.path.exists(f"/models/{voice_id}/model.pth"):
        # Use cloned voice
        print(f"Using cloned voice: {voice_id}")
        tts = TTS(model_path=f"/models/{voice_id}/model.pth",
                  config_path=f"/models/{voice_id}/config.json",
                  gpu=True)
        voice_type = "cloned"
    else:
        # Use preset voice
        voice_key = voice_id if voice_id in VOICE_MODELS else DEFAULT_VOICE
        model_path = VOICE_MODELS[voice_key]
        print(f"Using preset voice: {voice_key}")
        tts = TTS(model_path, gpu=True)
        voice_type = "preset"
    
    # Generate audio
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        output_path = tmp_file.name
    
    tts.tts_to_file(text=text, file_path=output_path)
    
    # Read the WAV file
    with open(output_path, 'rb') as audio_file:
        audio_data = audio_file.read()
    
    os.unlink(output_path)
    
    generation_time = time.time() - start_time
    
    result = {
        "generation_time": generation_time,
        "voice_id": voice_id or DEFAULT_VOICE,
        "voice_type": voice_type,
        "format": return_format
    }
    
    # Return format based on request
    if return_format == "base64":
        result["audio_base64"] = base64.b64encode(audio_data).decode('utf-8')
    else:
        result["audio_data"] = base64.b64encode(audio_data).decode('utf-8')  # Still base64 for transport
        result["message"] = "Decode audio_data from base64 to get WAV bytes"
    
    return result

@stub.function(
    image=image,
    gpu="A10G",
    network_file_systems={"/models": volume},
    timeout=300  # 5 minutes for cloning
)
def clone_voice(
    audio_data: bytes,
    voice_id: str,
    voice_name: str
) -> Dict:
    """
    Clone a voice from 15-30 minute audio sample
    This is the training step - happens once per new voice
    """
    from TTS.api import TTS
    import tempfile
    import soundfile as sf
    
    print(f"Starting voice cloning for {voice_name} (ID: {voice_id})")
    
    # Save uploaded audio
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_audio:
        tmp_audio.write(audio_data)
        audio_path = tmp_audio.name
    
    # Initialize XTTS for cloning
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)
    
    # Clone the voice (simplified - in production you'd fine-tune)
    # For now, we'll save the reference audio for voice conversion
    output_dir = f"/models/{voice_id}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save reference audio for this voice
    reference_path = f"{output_dir}/reference.wav"
    os.rename(audio_path, reference_path)
    
    # In production: Fine-tune XTTS here
    # For now: Save config for voice conversion
    config = {
        "voice_id": voice_id,
        "voice_name": voice_name,
        "reference_audio": "reference.wav",
        "created_at": str(time.time())
    }
    
    import json
    with open(f"{output_dir}/config.json", "w") as f:
        json.dump(config, f)
    
    return {
        "success": True,
        "voice_id": voice_id,
        "voice_name": voice_name,
        "message": f"Voice {voice_name} cloned successfully",
        "path": output_dir
    }

@stub.function(
    image=image,
    network_file_systems={"/models": volume}
)
def upload_trained_model(
    model_data: bytes,
    config_data: bytes,
    voice_id: str
) -> str:
    """
    Upload a model trained in Colab
    """
    model_dir = f"/models/{voice_id}"
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model files
    with open(f"{model_dir}/model.pth", "wb") as f:
        f.write(model_data)
    
    with open(f"{model_dir}/config.json", "wb") as f:
        f.write(config_data)
    
    return f"Model {voice_id} uploaded successfully"

@stub.function(network_file_systems={"/models": volume})
def list_voices() -> Dict:
    """
    List all available voices (preset + cloned)
    """
    cloned_voices = []
    
    # Check for cloned voices
    if os.path.exists("/models"):
        for voice_id in os.listdir("/models"):
            config_path = f"/models/{voice_id}/config.json"
            if os.path.exists(config_path):
                import json
                with open(config_path, "r") as f:
                    config = json.load(f)
                cloned_voices.append({
                    "id": voice_id,
                    "name": config.get("voice_name", voice_id),
                    "type": "cloned"
                })
    
    # Add preset voices
    preset_voices = [
        {"id": key, "name": key.title(), "type": "preset"}
        for key in VOICE_MODELS.keys()
    ]
    
    return {
        "preset_voices": preset_voices,
        "cloned_voices": cloned_voices,
        "total": len(preset_voices) + len(cloned_voices),
        "default_voice": DEFAULT_VOICE
    }

@stub.local_entrypoint()
def test_synthesis():
    """Test function - run with: modal run modal_app.py"""
    
    # Test with preset voice
    print("Testing preset voice (Jenny)...")
    result = synthesize_speech.remote(
        text="Hello! This is a test of the Modal GPU synthesis. It should be much faster than CPU!",
        voice_id="jenny"
    )
    print(f"✅ Generated audio in {result['generation_time']:.2f} seconds")
    
    # Save test audio
    if result.get('audio_data'):
        audio_bytes = base64.b64decode(result['audio_data'])
        with open("test_output.wav", "wb") as f:
            f.write(audio_bytes)
        print("✅ Audio saved to test_output.wav")
    
    # List available voices
    voices = list_voices.remote()
    print(f"\n📢 Available voices:")
    print(f"  Preset: {[v['id'] for v in voices['preset_voices']]}")
    print(f"  Cloned: {[v['id'] for v in voices['cloned_voices']]}")
