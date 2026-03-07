import requests
import json
import psutil

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.2:3b"

def check_if_running(process_name='ollama'):
    running = False
    for proc in psutil.process_iter(['name']):
        if process_name in proc.info['name']:
            running = True
            break
    return running

def call_ollama(messages, model=MODEL, OLLAMA_URL=OLLAMA_URL, temperature=0.8):
    payload = {
        'model': model,
        'messages': messages,
        'stream': False,
        'options': {
            'temperature': temperature,
            'top_p': 0.9
        }
    }

    r = requests.post(OLLAMA_URL, json=payload, timeout=240)
    r.raise_for_status()
    return r.json()["message"]["content"]