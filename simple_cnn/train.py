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
        if criterion == 'CrossEntropyLoss':
            self.criterion = nn.CrossEntropyLoss()
        # Add more conditions here for other loss functions as needed
        
        # Initialize optimizer
        if optimizer == 'SGD':
            self.optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
        elif optimizer == 'Adam':
            self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        # Add more conditions here for other optimizers as needed

    def train(self, epochs):
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