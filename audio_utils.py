import queue
import collections
import numpy as np
import torch
import torchaudio
import io
import webrtcvad
import sounddevice as sd

def record_utterance(sample_rate=16000, frame_duration=30):
    vad = webrtcvad.Vad(3) 
    frame_size = int(sample_rate * frame_duration / 1000)
    audio_q = queue.Queue()
    
    def audio_callback(indata, frames, time_info, status):
        audio_q.put(bytes(indata))
    
    with sd.RawInputStream(samplerate=sample_rate, blocksize=frame_size,
                         dtype='int16', channels=1, callback=audio_callback):
        ring_buffer = collections.deque(maxlen=5)  
        voiced_frames = []
        speech_detected = False
        silence_duration = 0
        required_silence = 0.7 
        
        while True:
            frame = audio_q.get()
            is_speech = vad.is_speech(frame, sample_rate)
            
            if is_speech:
                if not speech_detected:
                    print("...", flush=True)
                    speech_detected = True
                    voiced_frames.extend(ring_buffer) 
                voiced_frames.append(frame)
                silence_duration = 0
            else:
                if speech_detected:
                    silence_duration += frame_duration / 4000.0
                    voiced_frames.append(frame) 
                    
                    if silence_duration >= required_silence:
                        return b''.join(voiced_frames)
                    else:
                        ring_buffer.append(frame)

def transcribe_audio(model, audio_bytes):
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
    waveform = torch.from_numpy(audio_np.astype(np.float32) / 32768.0).unsqueeze(0)

    buffer = io.BytesIO()
    torchaudio.save(buffer, waveform, 16000, format='wav')
    buffer.seek(0)

    segments, _ = model.transcribe(buffer)
    text = " ".join([seg.text for seg in segments]).strip()
    print(f"You: {text}")
    return text