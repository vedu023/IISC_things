import os
from tqdm import tqdm
import torch as t
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split 
 

device = t.device('cuda' if t.cuda.is_available() else 'cpu')

train_path = 'AIP_assignments/a1/assignment1_data/train'
test_path = 'AIP_assignments/a1/assignment1_data/test'

batch_size = 2
num_epochs = 12
lr = 0.001
num_classes = len(os.listdir(train_path))

data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

train_data = datasets.ImageFolder(train_path, data_transform)
test_dataset = datasets.ImageFolder(test_path, data_transform)

train_dataset, val_dataset = random_split(train_data, [0.8, 0.2])

# Training and Validation Data Loaders
train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
val_dataloader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)
test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)


def init_weights(m):
    if type(m) == nn.Conv2d:
        t.nn.init.normal_(m.weight, mean=0.0, std=0.008)
        
model = models.resnet50(weights = models.ResNet50_Weights.DEFAULT) # remove weights for training from scratch...
num_features = model.fc.in_features
model.fc = t.nn.Linear(num_features, num_classes) 
model = model.to(device)
model.apply(init_weights)      # apply the initial weights for training from scratch
optimizer = t.optim.Adam(model.parameters(), lr = lr)
criterion = nn.CrossEntropyLoss()

# training...
for i in range(num_epochs):

    correct = 0
    total = 0
    for img, label in tqdm(train_dataloader):
        img = img.to(device)
        label = label.to(device)

        optimizer.zero_grad()

        outputs = model(img)
        loss = criterion(outputs, label)

        _, predicted = t.max(outputs.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()

        loss.backward()
        optimizer.step()
    print(f'Epoch [{i+1}/{num_epochs}], Loss: {loss.item():.4f} \nTraining Accuracy: {100 * correct / total:.2f}')

    # validation...
    with t.no_grad():
        correct = 0
        total = 0
        for images, labels in val_dataloader:
             
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = t.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f'Validation Accuracy: {100 * correct / total:.2f}%')
    print('\n','--'*50)

# Save the trained model...
t.save(model, 'model.pt')

## testing...
model = t.load('model.pt')
model.eval()

with t.no_grad():
    correct = 0
    total = 0
    for images, labels in tqdm(test_dataloader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = t.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('testing Accuracy: {:.2f}%'.format(100 * correct / total))