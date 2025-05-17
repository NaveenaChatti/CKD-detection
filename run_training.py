# run_training.py

print("[DEBUG] Running training script...")

from src.train_model import train

if __name__ == "__main__":
    print("[DEBUG] Starting training function...")
    train("data/kidney_disease.csv")
    print("[DEBUG] Training function complete.")
