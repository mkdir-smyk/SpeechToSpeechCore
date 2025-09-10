import torch
import sounddevice as sd

def init_tts():
    tts_model = torch.hub.load('snakers4/silero-models', 'silero_tts', 
                             language='en', speaker='v3_en')[0]
    return tts_model

def speak(text, tts_model, speaker='en_99', sample_rate=8000):
    try:
        audio = tts_model.apply_tts(text=text, 
                                  speaker=speaker, 
                                  sample_rate=sample_rate)
        sd.play(audio, sample_rate)
        sd.wait()
    except Exception as e:
        print(f"TTS Error: {e}")