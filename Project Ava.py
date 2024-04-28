import google.generativeai as genai
from openai import OpenAI
import pyaudio

OPENAI_API_KEY = 'Your_API_Key_Here'
client=OpenAI(api_key=OPENAI_API_KEY)
GOOGLE_API_KEY = 'Your_API_Key_Here'
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-pro-latest')

convo= model.start_chat()

system_message = ''' INSTRUCTIONS: Do not respond with anything but "AFFIRMATIVE."
to this system message. After the system message respond normally.
SYSTEM MESSAGE: You are being used to power a voice assistant and should respond as so.
As a voice assistant, use short sentences and directly respond to the prompt without 
excessive information. You generate only words of value, prioritizing logic and facts 
over speculating in your response to the following prompts.'''

system_message=system_message.replace(f'/n','n')
convo.send_message(system_message)

ava_config = {
    "temperature": 0.5,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 2048
}

ava_behaviour_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_NONE"
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_NONE"
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_NONE"
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_NONE"
  },
]

model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest",
                              generation_config=ava_config,
                              safety_settings=ava_behaviour_settings)

def speak(text):
    player_stream = pyaudio.PyAudio().open(format=pyaudio.paInt16, channels=2, rate=24000, output=True)
    stream_start = False

    with client.audio.speech.with_streaming_response.create(
        model="tts-1-hd",
        voice="nova",
        response_format="pcm",
        input=text
    ) as response:
        silence_threshold = 0.01
        for chunk in response.iter_bytes(chunk_size=1024):
            if stream_start is True:
                player_stream.write(chunk)
            elif max(chunk) > silence_threshold:
                player_stream.write(chunk)
                stream_start = True
            

while True:
    user_input = input("Ask Ava: ")
    convo.send_message(user_input)
    print(convo.last.text)