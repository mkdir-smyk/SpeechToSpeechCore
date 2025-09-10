import json
import requests
from tts_utils import speak

def generate_response(prompt, tts_model, conversation):

        formatted_prompt = conversation.format_prompt(prompt)
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            headers={"Content-Type": "application/json"},
            data=json.dumps({
                "model": "mistral",
                "prompt": formatted_prompt,
                "stream": True,
                "options": {
                    "temperature": 0.85,
                    "max_tokens": 150,
                }
            }),
            stream=True
        )
        
        response.raise_for_status()
        
        full_response = ""
        buffer = ""
        sentence_end_chars = {'.', '?', '!', ',', ';', ':'}
        
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line.decode('utf-8'))
                token = chunk.get("response", "")
                
                full_response += token
                buffer += token

                if token in sentence_end_chars or token.isspace():
                    if buffer.strip():
                        speak(buffer, tts_model, speaker='en_99')
                        buffer = ""
        
        if buffer.strip():
            speak(buffer, tts_model, speaker='en_99')
        
        print(f"AI: {full_response.strip()}")
        conversation.context.append(("assistant", full_response.strip(), {"sentiment": "neutral"}))
        
        return full_response.strip()
    