import ollama
import pyttsx3
import whisper
import sounddevice as sd
import numpy as np
from duckduckgo_search import DDGS
from resemblyzer import VoiceEncoder , preprocess_wav
import time

engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)
engine.setProperty('rate', 150)

encoder = VoiceEncoder()
my_voiceprint = np.load("voiceprint.npy")

personality = "You are WEDNESDAY - a highly intelligent, professional and reliable personal assistant that aims to please. You are concise and accurate, going out of your way to do and research things for the user, which you will address as Ayesha. You speak in a calm, clear and precise manner and dont waste words when responding to the user. You have access to real-time information but you never reference them or mention that you've searched for them - you speak and present your answer naturally as if you already know the answer whether you've needed to search or not."

def speak(text):
    engine.say(text)
    engine. runAndWait()

def listen():
    print("Listening...")
    recording = sd.rec(int(5 * 16000), samplerate = 16000 , channels = 1, dtype = 'float32')
    sd.wait()

    audio = recording.squeeze()
    wav = preprocess_wav(audio, source_sr = 16000)
    incoming_voiceprint = encoder.embed_utterance(wav)
    similarity = np.dot(my_voiceprint, incoming_voiceprint) / (np.linalg.norm(my_voiceprint) * np.linalg.norm(incoming_voiceprint))

    if similarity < 0.75:
        print("Voice Unidentified")
        return None
    
    model = whisper.load_model("base")
    result = model.transcribe(recording)
    return result ["text"]

def search(query):
    results = DDGS().text(query, max_results = 4)
    search_text = ""
    for result in results:
        search_text += result["title"] + "\n"
        search_text += result["body"] + "\n\n"
    return search_text

history = []

while True:
    user_input = listen() 
    print("You: " + user_input)

    if user_input is None:
        continue

    if user_input.lower() == "bye bye":
        break

    check = ollama.chat(model = "llama3" , messages=[
        {"role": "system", "content": "You decide if a question needs a real time web search to answer accurately. Reply with only 'yes' or 'no'. No other words"},
        {"role": "user", "content": user_input}
    ])

    needs_search = check["message"]["content"].strip().lower()

    if needs_search == "yes":
        search_results = search(user_input)
        user_message = user_input + "\n\nHere are some search results that may help answer this:\n" + search_results

    else:
        user_message = user_input
    
    history.append({"role" : "user", "content": user_message})
    response = ollama.chat(model = "llama3", messages = [{"role": "system", "content": personality}] + history)
    wednesday_response = response["message"]["content"]
    history.append({"role": "assistant", "content": wednesday_response})
    print("W.E.D.N.E.S.D.A.Y: " + wednesday_response)
    speak(wednesday_response)
