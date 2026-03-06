import torch
import torch.nn as nn
import timm  # This library holds the pre-trained Transformer models

class FloodViT(nn.Module):
    def __init__(self, num_classes=10, pretrained=True):
        super(FloodViT, self).__init__()
        
        # 1. Load the "Brain" (Vision Transformer)
        # We use 'vit_base_patch16_224'. It splits the image into 16x16 patches.
        print(f"🏗️ Loading Vision Transformer (Pretrained={pretrained})...")
        self.backbone = timm.create_model(
            'vit_base_patch16_224', 
            pretrained=pretrained, 
            num_classes=0 # Remove the original head (we will build our own)
        )
        
        # 2. Get the output size of the transformer (usually 768)
        self.embed_dim = self.backbone.num_features
        
        # 3. The "Classifier Head"
        # This part takes the transformer's thoughts and decides which of the 10 classes it is.
        self.head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, 512),
            nn.ReLU(),            # Activation function (adds non-linearity)
            nn.Dropout(0.3),      # Prevents the model from just memorizing images
            nn.Linear(512, num_classes) # Final output: 10 scores (one for each class)
        )

    def forward(self, x):
        # x is the input image [Batch_Size, 3, 224, 224]
        
        # Pass image through the Transformer
        features = self.backbone(x) 
        
        # Pass features through the Classifier
        logits = self.head(features)
        
        return logits

# --- TEST CODE (Runs only if you run this file directly) ---
if __name__ == "__main__":
    # Create a fake random image to test the plumbing
    dummy_img = torch.randn(1, 3, 224, 224) # [1 image, 3 channels (RGB), 224x224 pixels]
    
    # Initialize the model
    model = FloodViT(num_classes=10)
    
    # Try to make a prediction
    output = model(dummy_img)
    
    print("\n✅ SUCCESS: The Vision Transformer is built and running!")
    print(f"Input Image Shape: {dummy_img.shape}")
    print(f"Model Output Shape: {output.shape} (Should be [1, 10])")