"""
Quick test script to verify Modal deployment
Run with: python test_local.py
"""

import modal
import json

def test_modal_synthesis():
    # Look up your deployed function
    f = modal.Function.lookup("voice-synthesis-fast", "synthesize_speech")
    
    # Test synthesis
    result = f.remote(
        text="Testing Modal GPU synthesis. This should be fast!",
        voice_id="default"
    )
    
    print(f"✅ Synthesis completed in {result['generation_time']:.2f} seconds")
    print(f"✅ Audio generated: {len(result['audio_base64'])} bytes")
    
    # Save test audio
    import base64
    audio_data = base64.b64decode(result['audio_base64'])
    with open("test_output.wav", "wb") as f:
        f.write(audio_data)
    print("✅ Audio saved to test_output.wav")

if __name__ == "__main__":
    test_modal_synthesis()
