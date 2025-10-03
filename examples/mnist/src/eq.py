import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from safetensors.torch import save_file

# Hyperparameters
LR = 5e-4
BATCH_SIZE = 60
EPOCHS = 10
NORMALIZE = 255.0

# Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x / NORMALIZE)
])

train_dataset = datasets.MNIST(
    '.', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = datasets.MNIST(
    '.', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# Model
class MnistModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(1600, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, train=True):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        if train:
            x = self.dropout(x)
        x = self.fc(x)
        return x


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = MnistModel().to(device)
optimizer = optim.AdamW(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
    train_loss = total_loss / len(train_dataset)
    train_acc = correct / len(train_dataset)

    # Test
    model.eval()
    correct_test = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x, train=False)
            preds = logits.argmax(dim=1)
            correct_test += (preds == y).sum().item()
    test_acc = correct_test / len(test_dataset)

    print(f"Epoch {epoch+1}: Train loss={train_loss:.4f}, Train acc={
          train_acc:.4f}, Test acc={test_acc:.4f}")

state_dict = model.state_dict()  # get all weights as a dict of tensors
save_file(state_dict, "mnist_model.safetensors")
print("Model saved to mnist_model.safetensors")
