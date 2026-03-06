import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import FloodViT  # Import the model you just built
import os
from tqdm import tqdm # Progress bar

# --- CONFIGURATION ---
# Since you are on a CPU (Laptop), we keep batch size small to prevent crashing
BATCH_SIZE = 16 
LR = 0.0001        # Learning Rate (how fast it learns)
EPOCHS = 1         # We start with 1 epoch just to test. (Increase to 5 later)
DATA_DIR = "data/processed/classification"

def train():
    # 1. Setup Device (Use CPU)
    device = torch.device("cpu")
    print(f"⚙️ Training on device: {device}")

    # 2. Prepare Data Transforms
    # The model expects 224x224 pixel images
    print("📂 Loading EuroSAT Data...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 3. Load Datasets
    train_path = os.path.join(DATA_DIR, 'train')
    test_path = os.path.join(DATA_DIR, 'test')
    
    # Check if data exists
    if not os.path.exists(train_path):
        print(f"❌ Error: Could not find training data at {train_path}")
        print("Did you run 'setup_data.py' successfully?")
        return

    train_data = datasets.ImageFolder(train_path, transform=transform)
    test_data = datasets.ImageFolder(test_path, transform=transform)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"✅ Data Loaded: {len(train_data)} training images | {len(test_data)} test images.")

    # 4. Initialize Model
    model = FloodViT(num_classes=10).to(device)
    
    # 5. Define Training Tools (Loss & Optimizer)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 6. Training Loop
    print(f"\n🚀 Starting Training for {EPOCHS} Epoch(s)...")
    print(" (Note: This might take 10-20 minutes on a laptop CPU. Be patient!)")
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Create a progress bar
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass (Guess)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass (Learn)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Calculate Accuracy for this batch
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            loop.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100 * correct / total:.2f}%")

        # Save the model after training
        print(f"✅ Epoch {epoch+1} Complete! Accuracy: {100 * correct / total:.2f}%")
        torch.save(model.state_dict(), "flood_model.pth")
        print("💾 Model saved to 'flood_model.pth'")

if __name__ == "__main__":
    train()