import torch
import torch.nn as nn
import torch.optim as optim

class Trainer:
    def __init__(self, model, device, train_dl, criterion='CrossEntropyLoss', optimizer='SGD', learning_rate=0.01, momentum=0.0):
        self.model = model
        self.device = device
        self.train_dl = train_dl
        self.learning_rate = learning_rate
        self.momentum = momentum
        
        # Initialize criterion
        self.criterion = self._init_criterion(criterion)
        
        # Initialize optimizer
        self.optimizer = self._init_optimizer(optimizer)

    def _init_criterion(self, criterion):
        if criterion == 'CrossEntropyLoss':
            return nn.CrossEntropyLoss()
        elif criterion == 'MSELoss':
            return nn.MSELoss()
        elif criterion == 'NLLLoss':
            return nn.NLLLoss()
        # Add more loss functions as needed
        else:
            raise ValueError(f"Unsupported loss function: {criterion}")

    def _init_optimizer(self, optimizer):
        if optimizer == 'SGD':
            return optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum)
        elif optimizer == 'Adam':
            return optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # Add more optimizers as needed
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")

    def train(self, epochs):
        self.model.to(self.device)
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for images, labels in self.train_dl:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f'Epoch {epoch+1}, Loss: {total_loss / len(self.train_dl)}')
