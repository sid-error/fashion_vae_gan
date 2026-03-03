import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_training_results(csv_file='experiment_results.csv'):
    # Load the logged data from your RTX 3050 training run
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print("CSV file not found. Ensure you have run train.py first.")
        return

    # Set the style for professional reports
    sns.set_theme(style="whitegrid")
    
    # Create a multi-panel plot to see all 10-class metrics clearly
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Panel 1: Structural & Latent Losses
    ax1.plot(df['Epoch'], df['Recon_Loss'], label='Reconstruction (Pixel Accuracy)', color='blue', linewidth=2)
    ax1.plot(df['Epoch'], df['KL_Loss'], label='KL Divergence (Latent Regularization)', color='green', linestyle='--')
    ax1.set_title('VAE Structural Training Components', fontsize=14)
    ax1.set_ylabel('Loss Value')
    ax1.legend()
    
    # Panel 2: Adversarial & Shape Recognition Losses
    ax2.plot(df['Epoch'], df['Adv_Loss'], label='Adversarial (Realism)', color='red', linewidth=2)
    ax2.plot(df['Epoch'], df['Feat_Loss'], label='Feature Matching (Shape Consistency)', color='purple', linewidth=2)
    ax2.set_title('GAN Adversarial & Shape Recognition Components', fontsize=14)
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('Loss Value')
    ax2.legend()
    
    plt.tight_layout()
    
    # Save the plot for your final project report
    plt.savefig('perfected_training_curves.png', dpi=300)
    print("Visualization saved as perfected_training_curves.png")
    plt.show()

if __name__ == "__main__":
    plot_training_results()