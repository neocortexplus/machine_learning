from data_loader import get_dataloaders
from model import CNN
from train import Trainer  # Assuming Trainer is your training class
from evaluate import Evaluator  # Make sure to import the Evaluator class
from config import get_config
import torch

def main():
    config = get_config()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dl, test_dl = get_dataloaders(batch_size=config.batch_size)
    model = CNN().to(device)
    
    # Assuming you have a Trainer class for training
    trainer = Trainer(model=model, device=device, train_dl=train_dl, criterion='CrossEntropyLoss', optimizer='SGD', learning_rate=config.learning_rate)
    trainer.train(epochs=config.epochs)
    
    # Initialize and use the evaluator
    evaluator = Evaluator(model=model, device=device, test_dl=test_dl)
    evaluator.evaluate()

if __name__ == "__main__":
    main()
