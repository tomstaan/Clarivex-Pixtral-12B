from flask import Flask, request, jsonify, abort
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import base64
import io
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Bearer token for authentication, fetched from .env file
BEARER_TOKEN = os.getenv("BEARER_TOKEN")

# Load the fine-tuned model and processor
model_id = "./fine-tuned-model/checkpoint-15"  # or "/app/fine-tuned-model/checkpoint-15" depending on the solution
model = LlavaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map='auto')
processor = AutoProcessor.from_pretrained(model_id)

# Function to validate the token
def validate_token(request):
    auth_header = request.headers.get('Authorization')
    if auth_header is None:
        abort(401, "Missing Authorization header")
    token = auth_header.split(" ")[1]
    if token != BEARER_TOKEN:
        abort(403, "Invalid Bearer token")

@app.route('/predict', methods=['POST'])
def predict():
    # Check for valid authentication
    validate_token(request)

    # Get the image and question from the request
    data = request.json
    image_data = data.get('image')
    question = data.get('question')

    if not image_data or not question:
        abort(400, "Both 'image' and 'question' are required")

    # Convert image data from base64 to PIL Image
    image = Image.open(io.BytesIO(base64.b64decode(image_data)))

    # Process input through the model
    inputs = processor(text=[question], images=[image], return_tensors="pt", padding=True).to('cuda')

    # Generate response
    generated_ids = model.generate(**inputs, max_new_tokens=64)
    generated_text = processor.batch_decode(generated_ids[:, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)[0]

    # Return the prediction as a JSON response
    return jsonify({"response": generated_text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

