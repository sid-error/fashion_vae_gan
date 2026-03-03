# gENERATIVE AI PROJECT (23CSE475)
# SIDHARTH S NAIR : CB.SC.U4CSE23443

# Fashion Designer AI: Conditional VAE-GAN

A high-performance **Conditional Variational Autoencoder Generative Adversarial Network (CVAE-GAN)** designed to generate high-fidelity, category-specific fashion designs. This project leverages **Spectral Normalization**, **Feature Matching Loss**, and the **Two-Time Scale Update Rule (TTUR)** to ensure stable training and recognizable outputs across all 10 Fashion-MNIST classes.

---

## Key Technical Features

* **Conditional Generation**: Uses label embeddings to provide class-specific blueprints to both the Encoder and Generator, allowing for targeted design creation.
* **Architectural Stability**: Implements **Spectral Normalization** in the Discriminator to constrain weight gradients and prevent mode collapse.
* **Structural Fidelity**: Employs **Feature Matching Loss**, forcing the Generator to match high-level internal activations of the Discriminator for consistent shape recognition.
* **Stable Upsampling**: Utilizes **Resize-Convolution** to eliminate checkerboard artifacts and produce smooth textures.

---

## Tech Stack

* **Backend**: Python, PyTorch (CUDA enabled)
* **Web Framework**: Flask
* **Frontend**: HTML5, CSS3, JavaScript
* **Hardware Target**: NVIDIA RTX 3050 (8GB VRAM optimized)

---

## Project Structure

```text
Fashion_VAE_GAN/
├── app.py                  # Flask web application & Inference server
├── train.py                # Main conditional training loop (100 Epochs)
├── evaluator.py            # Scientific metric calculation (MAE, Disc Accuracy)
├── inference.py            # Standalone script for generating class-specific grids
├── data_loader.py          # Data pipeline for Fashion-MNIST (10 Classes)
├── loss_functions.py       # Hybrid Loss (VAE + GAN + Feature Matching)
├── visualise_results.py    # Training curve visualization engine
├── models/
|   ├── __init__.py
│   ├── vae_gan_wrapper.py      # Wrapper for coordinated VAE-GAN forward passes
│   ├── vae_encoder.py          # Conditional Encoder
│   ├── vae_gan_generator.py    # Conditional Generator with smooth upsampling
│   └── gan_discriminator.py    # Spectral Norm Discriminator
├── templates/
│   └── index.html          # Web interface for design generation
└── checkpoints/            # Directory for saved model weights (.pth)
└── data/                   # FashionMNIST Dataset
```

---

## Setup & Installation

Create a virtual environment and install dependencies:
python -m venv venv
.\venv\Scripts\activate
pip install torch torchvision pandas matplotlib seaborn flask pillow scikit-learn

To train the model from scratch for 100 epochs on your GPU:
python train.py
Logs: Metrics are saved in experiment_results.csv.
Checkpoints: Saved every 10 epochs in /checkpoints.

---

## Testing

Run the evaluator to calculate Mean Absolute Error (MAE) and Discriminator Accuracy to verify GAN equilibrium:
python evaluator.py

Generate a 4x4 grid of specific fashion items (e.g., Bags or Dresses) to verify visual quality:
python inference.py

Launch the local web server to generate designs via the UI:
python app.py
Access the UI at: http://127.0.0.1:5000
Select a category and click Generate New Design.