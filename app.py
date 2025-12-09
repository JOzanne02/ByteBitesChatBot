from flask import Flask, request, jsonify
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import os

# --- Model Loading (Keep this outside the request handling for speed) ---
app = Flask(__name__)

# Set up model and tokenizer globally
model = None
tokenizer = None
model_name = "flax-community/t5-recipe-generation"

def load_model():
    """Loads the model and tokenizer."""
    global model, tokenizer
    print("🔄 Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Check if a GPU is available, otherwise use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        dtype=torch.float32 if device == "cpu" else torch.float16, # Use float32 on CPU
        device_map=device
    )
    tokenizer.pad_token = tokenizer.eos_token
    print(f"✅ Model loaded: {model_name} on device: {device}")

# --- Chat Function (Your original logic, adapted slightly) ---
def chat_with_bot(user_input, conversation_history):
    global model, tokenizer
    # ... [Paste your existing chat_with_bot function here] ...
    # (Make sure to remove the `model.device` and use the actual device from the loaded model if device_map="auto" is used, or pass the device)
    # Since we use device_map="auto", model will manage it, so the original logic should largely work.
    # Just ensure `inputs = {k: v.to(model.device) for k, v in inputs.items()}` is correct.

    # Build conversation context
    prompt = ""
    for turn in conversation_history[-6:]: 
        if turn['role'] == 'user':
            prompt += f"User: {turn['text']}\n"
        else:
            prompt += f"Assistant: {turn['text']}\n"

    prompt += f"User: {user_input}\nAssistant:"

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)

    # Move inputs to the correct device
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.3
        )

    # Decode only the new tokens
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    # Clean up response
    response = response.split("User:")[0].strip()
    response = response.split("\n\n")[0].strip()

    return response


# --- Flask Routes ---

@app.route('/')
def index():
    """Serve the frontend HTML file."""
    # For simplicity, we'll serve the chat via a single route.
    return app.send_static_file('index.html') # Serve the frontend file (see step 2)

@app.route('/api/chat', methods=['POST'])
def handle_chat():
    """API endpoint to get the bot's response."""
    if not model or not tokenizer:
        return jsonify({'error': 'Model not loaded.'}), 503

    data = request.get_json()
    user_input = data.get('user_input', '').strip()
    history = data.get('history', []) # Expects a list of {'role': 'user'/'bot', 'text': '...'}

    if not user_input:
        return jsonify({'error': 'No input provided'}), 400

    try:
        bot_response = chat_with_bot(user_input, history)
        return jsonify({'response': bot_response})
    except Exception as e:
        app.logger.error(f"Chat error: {e}")
        return jsonify({'error': f'An internal error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    load_model() # Load the model once when the server starts
    # Host on http://127.0.0.1:5000/
    app.run(debug=True)
