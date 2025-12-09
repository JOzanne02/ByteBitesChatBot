# app.py

import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# --- 1. Model Loading (Cached for speed) ---

# Use Streamlit's caching mechanism to load the model only once
@st.cache_resource
def load_model():
    model_name = "flax-community/t5-recipe-generation"
    
    # Hugging Face Spaces typically runs on CPU/GPU hardware, 
    # but we will rely on the default device mapping (device_map="auto") 
    # for best performance in the cloud environment.
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="auto"
    )
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

model, tokenizer = load_model()


# --- 2. Chat Function ---
def chat_with_bot(user_input, conversation_history):
    # This function is exactly your original logic, adapted slightly for the structure
    
    # Build conversation context
    prompt = ""
    for turn in conversation_history[-6:]: 
        # Note: conversation_history uses the Streamlit format: {"role": "user"/"assistant", "content": "text"}
        role_label = "User" if turn['role'] == 'user' else "Assistant"
        prompt += f"{role_label}: {turn['content']}\n"

    prompt += f"User: {user_input}\nAssistant:"

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    # Move inputs to the correct device (model.device ensures it's on the correct hardware)
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

    # Decode
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    response = response.split("User:")[0].strip()
    response = response.split("\n\n")[0].strip()
    
    return response

# --- 3. Streamlit Interface ---

st.set_page_config(page_title="ByteBites ChefMate", page_icon="🍲")
st.title("🍲 ByteBites ChefMate AI")
st.caption("Powered by flax-community/t5-recipe-generation")

# Initialize chat history (using Streamlit's session_state)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input
if prompt := st.chat_input("What culinary adventure should we embark on?"):
    
    # 1. Display and save user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 2. Get and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("💭 ChefMate is thinking up a recipe..."):
            
            # Use the history (excluding the current prompt) for context in the model call
            history_for_context = st.session_state.messages[:-1] 
            response = chat_with_bot(prompt, history_for_context)
            
            st.markdown(response)
    
    # 3. Save assistant response
    st.session_state.messages.append({"role": "assistant", "content": response})
