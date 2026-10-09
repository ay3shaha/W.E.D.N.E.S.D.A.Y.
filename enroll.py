import sounddevice as sd
import numpy as np
from resemblyzer import VoiceEncoder , preprocess_wav
import time

encoder = VoiceEncoder()

print("Enrolling voice - speak for 5 seconds when prompted")
print("Starting in 3...")
time.sleep(3)

print("Speak:")
recording = sd.rec(int(5 * 16000) , samplerate = 16000 , channels = 1 , dtype = 'float32')
sd.wait()
print("All finished!")

audio = recording.squeeze()
wav = preprocess_wav(audio, source_sr = 16000)
embedding = encoder.embed_utterance(wav)
np.save("voiceprint.npy" , embedding)

print("Voiceprint is now ready to be used")