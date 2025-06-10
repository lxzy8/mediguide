import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import os

# --- Configuration ---
# IMPORTANT: This path MUST point to the directory where your
# 'adapter_config.json', 'adapter_model.safetensors',
# and the tokenizer files (vocab.json, merges.txt, etc.) are located.
# If your friend downloads the 'mediguide-falcon' folder, they should
# put this path to that folder.
# Example: If 'mediguide-falcon' is in the same directory as this script, set it to "./mediguide-falcon"
# Or if it's in a specific path like "C:/Users/Friend/Desktop/mediguide-falcon"
LOCAL_MODEL_DIR = "./mediguide-falcon" # <--- YOUR FRIEND SHOULD VERIFY/CHANGE THIS PATH!

# The base model name (used to load the original Falcon model)
BASE_MODEL_NAME = "tiiuae/falcon-rw-1b"

# --- Device Setup ---
# Check if a CUDA GPU is available, otherwise fall back to CPU.
# Note: For Apple Silicon (MPS), you would need 'mps' support,
# but bitsandbytes is primarily designed for CUDA GPUs.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --- Load Model and Tokenizer ---
try:
    print(f"Loading PEFT config from: {LOCAL_MODEL_DIR}")
    peft_config = PeftConfig.from_pretrained(LOCAL_MODEL_DIR)

    print(f"Initializing BitsAndBytesConfig...")
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0
    )

    print(f"Loading base model '{BASE_MODEL_NAME}' with 8-bit quantization...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        device_map="auto", # "auto" will try to use GPU if available, or split across devices
        quantization_config=bnb_config,
        trust_remote_code=True # Needed for Falcon models
    )

    print(f"Loading tokenizer for '{BASE_MODEL_NAME}'...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token # Ensure padding token is set

    print(f"Applying LoRA adapters from: {LOCAL_MODEL_DIR}")
    model = PeftModel.from_pretrained(base_model, LOCAL_MODEL_DIR)
    model.eval() # Set model to evaluation mode (important for inference)

    print("\nFine-tuned model and tokenizer loaded successfully!")

except Exception as e:
    print(f"\nError loading model or tokenizer: {e}")
    print("Please ensure:")
    print(f"1. The path '{LOCAL_MODEL_DIR}' is correct and contains all adapter and tokenizer files.")
    print("2. You have installed all required libraries: pip install -U transformers peft bitsandbytes accelerate")
    print("3. Your GPU (if using) drivers are up to date and bitsandbytes is compiled for your GPU.")
    exit() # Exit if model loading fails


# --- Inference Function ---
def get_medical_guidance(question: str, max_tokens: int = 250) -> str:
    """
    Generates a medical guidance response from the fine-tuned model.
    """
    # Format the input similar to how your training data was structured
    # (e.g., "Patient: [Patient_Answer]\nDoctor: [Doctor_response]")
    formatted_input = f"Patient: {question}\nDoctor:"

    # Tokenize the input and move it to the correct device
    inputs = tokenizer(formatted_input, return_tensors="pt").to(device)

    # Generate a response
    with torch.no_grad(): # Disable gradient calculations for inference
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            num_return_sequences=1,
            do_sample=True,          # Use sampling for more natural responses
            top_k=50,                # Consider only the top 50 most likely tokens
            top_p=0.95,              # Use nucleus sampling
            temperature=0.7,         # Controls randomness
            pad_token_id=tokenizer.eos_token_id # Important for ending generation cleanly
        )

    # Decode the generated tokens back into human-readable text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Attempt to extract just the answer part
    try:
        # Assuming the model generates "Patient: ... \nDoctor: [answer here]"
        # We want everything after the last "Doctor:"
        answer_part = generated_text.split("Doctor:")[-1].strip()
        # Remove any remaining "Patient:" if the model repeats the prompt
        if answer_part.startswith("Patient:"):
            answer_part = answer_part.split("Patient:", 1)[1].strip()
        return answer_part
    except IndexError:
        return "Error extracting answer. Full generated text: " + generated_text


# --- Interactive Testing Loop ---
print("\n--- MediGuide E-Doctor Test ---")
print("Type your medical question. Type 'exit' to quit.")

while True:
    user_input = input("\nYour Question: ").strip()
    if user_input.lower() == 'exit':
        break

    print("MediGuide E-Doctor: Generating response...")
    response = get_medical_guidance(user_input)
    print(response)

print("\nThank you for using MediGuide E-Doctor!")

