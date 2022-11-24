import speech_recognition as sr
# print(sr.__version__) #3.8.1

r=sr.Recognizer()
with sr.AudioFile('src05.wav') as source:
    audio=r.record(source, duration=240)
    
vtotext=r.recognize_google(audio_data=audio,language='ko-KR')
print(vtotext)


# src05 : 어찌 내가 왕이 될 상인가 어서 말해 보게 내가 왕이 될 상인가 말이야