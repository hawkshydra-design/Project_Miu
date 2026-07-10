# Google Genai tool calling with history 
### Uses a sample function
import yaml
import gradio as gr
import json
import os
from google import genai

with open('character_config.yaml', 'r') as f:
    char_config = yaml.safe_load(f)

client = genai.Client(api_key=char_config['GOOGLE_API_KEY'])

# Constants
HISTORY_FILE = char_config['history_file']
MODEL = char_config['model']
SYSTEM_PROMPT = char_config['presets']['default']['system_prompt']

# Load/save chat history
def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return []

def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)



def get_miu_response_no_tool(messages):

    # Call Google Genai with system prompt + history
    response = client.models.generate_content(
        model=MODEL,
        contents=messages,
        config={
            "system_instruction": SYSTEM_PROMPT,
            "temperature": 1,
            "top_p": 1,
            "max_output_tokens": 2048,
        },
    )

    return response


def llm_response(user_input):

    messages = load_history()

    # Append user message to memory
    messages.append({
        "role": "user",
        "parts": [{"text": user_input}]
    })


    miu_test_response = get_miu_response_no_tool(messages)


    # just append assistant message to regular response. 
    messages.append({
    "role": "model",
    "parts": [{"text": miu_test_response.text}]
    })

    save_history(messages)
    return miu_test_response.text


if __name__ == "__main__":
    print('running main')