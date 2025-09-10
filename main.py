import threading
import os
from audio_utils import record_utterance, transcribe_audio
from tts_utils import init_tts, speak
from conversation_manager import ConversationManager
from response_generator import generate_response
from faster_whisper import WhisperModel

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def main():
    tts_model = init_tts()
    stt_model = WhisperModel("tiny.en", device="cuda", compute_type="int8_float32")
    conversation = ConversationManager()

    speak("Hello, what do you have in mind today...?", tts_model)

    try:
        while True:
            audio = record_utterance()
            if not audio:
                continue
            
            user_text = transcribe_audio(stt_model, audio)
            if not user_text:
                continue
            
            bot_response = generate_response(user_text, tts_model, conversation)

    except KeyboardInterrupt:
        speak("Goodbye!", tts_model)

if __name__ == "__main__":
    main()