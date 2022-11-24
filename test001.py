import speech_recognition as sr
print(sr.__version__) #3.8.1

r=sr.Recognizer()
with sr.AudioFile('dst001.wav') as source:
    audio=r.record(source, duration=240)
    
vtotext=r.recognize_google(audio_data=audio) #,language='ko-KR')
print(vtotext)


# dst 001 : Chancellor just received a report from Master Kenobi has engaged General Grievous