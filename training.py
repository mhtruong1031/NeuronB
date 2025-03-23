import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from BrainSegDataset import BrainSetDataset
from preprocessing import PROCESSED_DATA_PATH

from Model import UNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # GPU init

# Load data
train_inputs  = torch.load(PROCESSED_DATA_PATH+'training_inputs.pt')
train_outputs = torch.load(PROCESSED_DATA_PATH+'training_outputs.pt')
test_inputs   = torch.load(PROCESSED_DATA_PATH+"testing_inputs")
test_outputs  = torch.load(PROCESSED_DATA_PATH+"testing_outputs")

training_data = BrainSetDataset(train_inputs, train_outputs)
testing_data = BrainSetDataset(test_inputs, test_outputs)

train_loader  = DataLoader(training_data, batch_size=32)
test_loader   = DataLoader(testing_data, batch_size=32, shuffle=False)

model = UNet(2, 1) # Initialize model
model = model.to(device)

# Hyperparameters
epochs        = 5
learning_rate = 0.001
batch_size    = 32

critereon = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def main() -> None:
    last_val_loss = 1e99
    for e in range(epochs):
        model.train() # Training mode
        train_loss = 0
        
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs) # Get output of model
            if targets.dim() == 3:
                targets = targets.unsqueeze(1)
            loss = critereon(outputs, targets) # Calculate loss

            optimizer.zero_grad() # Set gradient to 0
            loss.backward() # Backprop for loss
            optimizer.step() # Update parameters

            training_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        val_dice = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                if targets.dim() == 3:
                    targets = targets.unsqueeze(1)
                
                outputs = model(inputs)
                loss = critereon(outputs, targets)

                val_loss += loss.item()
                val_dice += dice_score(outputs, targets)
                
                avg_train_loss = train_loss / len(train_loader)
                avg_val_loss   = val_loss / len(test_loader)
                avg_val_dice   = val_dice / len(test_loader)
                
        if avg_val_loss < last_val_loss:
            last_val_loss = avg_val_loss
            torch.save(model.state_dict(), f'/kaggle/working/final_model_{e+1}.pth')
        
        print(f'Epoch [{e+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Dice Score: {avg_val_dice:.4f}')

def dice_score(preds, targets, threshold=0.5, eps=1e-6):
    preds = (torch.sigmoid(preds) > threshold).float()
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum()
    return (2. * intersection + eps) / (union + eps)

if __name__ == '__main__':
    main()


