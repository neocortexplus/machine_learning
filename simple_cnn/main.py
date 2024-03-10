from data_loader import get_dataloaders
from model import CNN
# Ensure Trainer class is imported instead of the train function
from train import Trainer 
from evaluate import evaluate
from config import get_config
import torch

def main():
    config = get_config()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data loaders
    train_dl, test_dl = get_dataloaders(batch_size=config.batch_size)
    
    # Initialize the model
    model = CNN().to(device)
    
    # Initialize the trainer with the desired configuration
    trainer = Trainer(model=model, 
                      device=device, 
                      train_dl=train_dl, 
                      criterion='CrossEntropyLoss', 
                      optimizer='SGD', 
                      learning_rate=config.learning_rate, 
                      momentum=config.get('momentum', 0.9)) # Example of how to handle optional config parameters
    
    # Start the training process
    trainer.train(epochs=config.epochs)
    
    # Evaluate the model after training
    evaluate(model, device, test_dl)

if __name__ == "__main__":
    main()