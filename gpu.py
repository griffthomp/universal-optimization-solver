import os
import openai
try:
    client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    models = client.models.list()
    available = [m.id for m in models.data if 'gpt-4' in m.id]
    print('Available GPT-4 models:', available)
except Exception as e:
    print(f'Error: {e}')