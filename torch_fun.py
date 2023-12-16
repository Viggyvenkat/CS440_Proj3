import numpy as np
import torch 
import torch.nn as nn
import torchvision 
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split

dataset_path = '/Users/vigneshvenkat/Desktop/AI-Project-3/dataset_2'
transform = torchvision.transforms.Compose([
    transforms.Resize((20,20)),
    transforms.ToTensor(),
])

dataset = ImageFolder(root=dataset_path, transform=transform)
TOTAL_SAMPLES = len(dataset)
NUM_EPOCHS = 20
learning_rate = 0.01

TRAIN_SAMPLES = int(0.7 * TOTAL_SAMPLES)
TEST_SAMPLES = TOTAL_SAMPLES - TRAIN_SAMPLES

train_dataset, test_dataset = random_split(dataset, [TRAIN_SAMPLES, TEST_SAMPLES])

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
num_classes = len(dataset.classes)


class LogisticRegressionModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = self.linear(x)
        return nn.functional.softmax(x, dim=1)

model = LogisticRegressionModel(TOTAL_SAMPLES, num_classes)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(NUM_EPOCHS):
    for inputs, labels in train_dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {loss.item()}')