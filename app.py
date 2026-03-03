import torch
import io
import base64
import os
from flask import Flask, render_template, request, jsonify
from PIL import Image
from models.vae_gan_wrapper import VAEGAN

app = Flask(__name__)

# Config: Deployment for RTX 3050 environment
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 128
num_classes = 10 
model = VAEGAN(latent_dim, num_classes).to(device)

def load_latest_checkpoint():
    """Dynamically finds and safely loads the latest available weights."""
    checkpoint_dir = 'checkpoints'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        return False

    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    if not checkpoints:
        print("No checkpoints found. Please train the model first.")
        return False
    
    # Select the highest epoch number available (e.g., epoch 100)
    latest = sorted(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))[-1]
    path = os.path.join(checkpoint_dir, latest)
    
    try:
        # security update: weights_only=True prevents arbitrary code execution
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Successfully loaded weights from: {path}")
        return True
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return False

# Initial load and evaluation mode setting
load_latest_checkpoint()
model.eval()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    # Receive category (0-9) from frontend dropdown
    category_id = int(data.get('category', 0)) 
    
    with torch.no_grad():
        # 1. Sample Style (Gaussian Noise)
        z = torch.randn(1, latent_dim).to(device)
        
        # 2. Assign Identity (Conditional Class Label)
        label = torch.tensor([category_id]).to(device)
        
        # 3. Conditional Generation via perfected Decoder/Generator
        generated_tensor = model.generator(z, label)
        
        # Squeeze to get [1, 28, 28] -> [28, 28]
        generated_img = generated_tensor.cpu().squeeze().numpy()
        
    # Denormalize: map [-1, 1] back to [0, 255] for standard image display
    generated_img = ((generated_img * 0.5 + 0.5) * 255).clip(0, 255).astype('uint8')
    img = Image.fromarray(generated_img)
    
    # Convert PNG to Base64 for transmission to the HTML frontend
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    return jsonify({'image': img_str})

if __name__ == '__main__':
    # threaded=False ensures the GPU context remains stable during debug restarts
    app.run(debug=True, threaded=False)