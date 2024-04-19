import google.generativeai as genai

GOOGLE_API_KEY = 'AIzaSyAL5xzHxg0hQubxI_hyooA5aQhZmPQ8y1k'
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.0-pro-latest')

response = model.generate_content(input('Ask Ava: '))
print(response)