import torch

class Evaluator:
    def __init__(self, model, device, test_dl):
        self.model = model
        self.device = device
        self.test_dl = test_dl

    def evaluate(self):
        self.model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for images, labels in self.test_dl:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f'Accuracy: {accuracy}%')
        return accuracy