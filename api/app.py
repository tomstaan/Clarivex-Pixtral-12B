import os
from flask import Flask, request, jsonify, abort
from transformers import LlavaForConditionalGeneration, AutoProcessor
import torch
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Path to the fine-tuned model checkpoint
model_path = "/home/admin/jacobwalker/clarivex_ai/fine-tuned-model/checkpoint-30"

# Load the model and processor from the specified checkpoint
try:
    model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,  # Adjust dtype if needed
        device_map='auto'
    )
    processor = AutoProcessor.from_pretrained(model_path)
    tokenizer = processor.tokenizer
    tokenizer.padding_side = "left"  # For Flash Attention compatibility

    print("Model and processor loaded successfully from checkpoint-30.")
except Exception as e:
    print(f"Error loading the model: {e}")
    abort(500, description="Model loading failed")

# Define a basic API route for testing
@app.route('/')
def index():
    return jsonify({"message": "API is working!"})

# Define a prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if not request.json or 'input_text' not in request.json:
        abort(400, description="Input text is required.")
    
    input_text = request.json['input_text']

    # Tokenize input and prepare for the model
    inputs = processor(text=[input_text], return_tensors="pt", padding=True).to('cuda')

    # Generate output from the model
    generated_ids = model.generate(**inputs, max_new_tokens=64)
    generated_texts = processor.batch_decode(generated_ids[:, inputs["input_ids"].shape[-1]:])

    return jsonify({"response": generated_texts[0]})

if __name__ == '__main__':
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000)

