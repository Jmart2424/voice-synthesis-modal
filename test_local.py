"""
Test both cloning and synthesis
"""

import modal
import requests
import base64
import time

def test_synthesis():
    """Test synthesis with preset and cloned voices"""
    f = modal.Function.lookup("voice-synthesis-fast", "synthesize_speech")
    
    # Test preset voice
    print("Testing preset voice (Jenny)...")
    result = f.remote(
        text="Testing Modal GPU synthesis. This should take 2-3 seconds!",
        voice_id="jenny",
        return_format="wav"
    )
    
    print(f"✅ Generated in {result['generation_time']:.2f} seconds")
    
    # Save audio
    audio_bytes = base64.b64decode(result['audio_data'])
    with open("test_jenny.wav", "wb") as file:
        file.write(audio_bytes)
    print("✅ Saved to test_jenny.wav")
    
    # List all voices
    list_f = modal.Function.lookup("voice-synthesis-fast", "list_voices")
    voices = list_f.remote()
    print(f"\n📢 Available voices: {voices['total']}")
    print(f"  Default: {voices['default_voice']}")

def test_api():
    """Test the HTTP API endpoints"""
    print("\nTesting HTTP API...")
    
    # Assuming you're running: modal serve deploy.py
    base_url = "http://localhost:8000"
    
    # Test synthesis endpoint (returns audio/wav)
    response = requests.post(
        f"{base_url}/synthesize",
        json={"text": "Testing API endpoint", "voice_id": "jenny"}
    )
    
    if response.status_code == 200:
        with open("test_api.wav", "wb") as f:
            f.write(response.content)
        print("✅ API test successful - saved test_api.wav")
        print(f"   Generation time: {response.headers.get('X-Generation-Time')}s")

if __name__ == "__main__":
    test_synthesis()
    # test_api()  # Uncomment if running modal serve
