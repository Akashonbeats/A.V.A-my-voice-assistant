import google.generativeai as genai

GOOGLE_API_KEY = 'You_API_Key_Here'
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.0-pro-latest')

response = model.generate_content(input('Ask Ava: '))
print(response.text)