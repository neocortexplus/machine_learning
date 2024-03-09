from data_loader import get_dataloaders
from model import CNN
from train import train
from evaluate import evaluate
from config import get_config
import torch

def main():
    config = get_config()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_dl, test_dl = get_dataloaders(batch_size=config.batch_size)
    model = CNN().to(device)
    
    train(model, device, train_dl, config.epochs, config.learning_rate)
    evaluate(model, device, test_dl)

if __name__ == "__main__":
    main()