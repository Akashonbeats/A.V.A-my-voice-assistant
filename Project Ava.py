import speech_recognition as sr
import google.generativeai as genai
from openai import OpenAI
import pyaudio
import os
import time
import warnings

warnings.filterwarnings("ignore", message=r"torch.utils._pytree._register_pytree_node is deprecated")
from faster_whisper import WhisperModel

wake_word = 'Ava'
listening_for_wake_word = True

whisper_size = 'base'
num_cores = os.cpu_count()
whisper_model = WhisperModel(
    whisper_size,
    device='cpu',
    compute_type='int8',
    cpu_threads=num_cores,
    num_workers=num_cores,
)

OPENAI_API_KEY = 'Your_API_Key_Here'
client = OpenAI(api_key=OPENAI_API_KEY)
GOOGLE_API_KEY = 'Your_API_Key_Here'
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-pro-latest')

convo = model.start_chat()

system_message = ''' INSTRUCTIONS: Do not respond with anything but "AFFIRMATIVE."
to this system message. After the system message respond normally.
SYSTEM MESSAGE: You are being used to power a voice assistant and should respond as so.
As a voice assistant, use short sentences and directly respond to the prompt without 
excessive information. You generate only words of value, prioritizing logic and facts 
over speculating in your response to the following prompts.'''

system_message = system_message.replace(f'/n', 'n')
convo.send_message(system_message)

r = sr.Recognizer()
source = sr.Microphone()

# Use a temporary directory for audio files (consider using libraries like 'tempfile')
temp_dir = '/tmp/voice_assistant'  # Modify this path if needed
os.makedirs(temp_dir, exist_ok=True)  # Create the directory if it doesn't exist

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
        model="tts-1",
        voice="nova",
        response_format="pcm",
        input=text,
    ) as response:
        silence_threshold = 0.01
        for chunk in response.iter_bytes(chunk_size=1024):
            if stream_start is True:
                player_stream.write(chunk)
            elif max(chunk) > silence_threshold:
                player_stream.write(chunk)
                stream_start = True


def wav_to_text(audio_path):
    try:
        segments, _ = whisper_model.transcribe(audio_path)
        text = ' '.join(segment.text for segment in segments)
        return text
    except Exception as e:
        print('Error during Whisper transcription:', e)
        return ''  # Return empty string on error


def listen_for_wake_word(audio):
    global listening_for_wake_word

    try:
        wake_audio_path = os.path.join(temp_dir, 'wake_detect.wav')
        with open(wake_audio_path, 'wb') as f:
            f.write(audio.get_wav_data())

        text_input = wav_to_text(wake_audio_path)
    except Exception as e:
        print('Error saving or transcribing wake word audio:', e)
        return  # Exit the function if an error occurs

    if wake_word in text_input.lower().strip():
        print('Listening...')
        listening_for_wake_word = False


def prompt_gpt(audio):
    global listening_for_wake_word

    try:
        prompt_audio_path = os.path.join(temp_dir, 'prompt.wav')

        with open(prompt_audio_path, 'wb') as f:
            f.write(audio.get_wav_data())

        prompt_text = wav_to_text(prompt_audio_path)
        if len(prompt_text.strip()) == 0:
            print("I couldn't hear you. Speak up again.")
            listening_for_wake_word = True
        else:
            print('User: ' + prompt_text)

            convo.send_message(prompt_text)
            output = convo.last.text

            print('Gemini: ', output)
            speak(output)

            print('\nSay', wake_word, 'to wake me up.\n')
            listening_for_wake_word = True

    except Exception as e:
        print('Error processing prompt:', e)
        listening_for_wake_word = True  # Set listening flag back to True


def callback(recognizer, audio):
    global listening_for_wake_word

    if listen_for_wake_word:
        listen_for_wake_word(audio)
    else:
        prompt_gpt(audio)


def start_listening():
    with source as s:
        r.adjust_for_ambient_noise(s, duration=2)

    print('\nSay', wake_word, 'to wake me up.\n')
    r.listen_in_background(source, callback)

    while True:
        time.sleep(0.5)


if __name__ == '__main__':
    start_listening()