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
