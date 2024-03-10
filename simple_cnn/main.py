from data_loader import DataLoaderFactory
from model import CNN
from train import Trainer  # Ensure Trainer class is ready as per previous discussion
from evaluate import Evaluator  # Make sure Evaluator class is defined
from config import get_config
import torch

def main():
    config = get_config()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize DataLoaderFactory with dataset name from config
    data_loader_factory = DataLoaderFactory(dataset_name=config.dataset_name, batch_size=config.batch_size)
    train_dl, test_dl = data_loader_factory.get_dataloaders()
    
    model = CNN().to(device)
    
    trainer = Trainer(model=model, device=device, train_dl=train_dl, criterion='CrossEntropyLoss', optimizer='SGD', learning_rate=config.learning_rate)
    trainer.train(epochs=config.epochs)
    
    evaluator = Evaluator(model=model, device=device, test_dl=test_dl)
    evaluator.evaluate()

if __name__ == "__main__":
    main()