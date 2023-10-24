import os, torch, warnings
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.models import mobilenet_v2

warnings.filterwarnings("ignore")

class MobileNetV2Encoder(nn.Module):
    def __init__(self):
        super(MobileNetV2Encoder, self).__init__()
        self.mobilenet = mobilenet_v2(pretrained=True)
        self.features = nn.Sequential(*list(self.mobilenet.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        return x

class MobileNetV2Decoder(nn.Module):
    def __init__(self, num_classes):
        super(MobileNetV2Decoder, self).__init__()
        self.conv1 = nn.Conv2d(1280, 256, 1)
        self.conv2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3 = nn.Conv2d(256, num_classes, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)
        x = F.relu(self.conv2(x))
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)
        x = self.conv3(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return x

class MobileNetV2Unet(nn.Module):
    def __init__(self, num_classes):
        super(MobileNetV2Unet, self).__init__()
        self.encoder = MobileNetV2Encoder()
        self.decoder = MobileNetV2Decoder(num_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class PASCAL_Dataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.images = os.listdir(os.path.join(root, 'Images'))
        
        self.labels = []
        self.n_images = []
        for image_name in self.images:
            label_name = os.path.join(self.root, 'Annotations', image_name[:-4] + '.png')
            if os.path.exists(label_name):
                self.labels.append(label_name)
                self.n_images.append(image_name)
        self.n_images.sort()
        self.labels.sort()
    
    
    def __getitem__(self, index):
    
        image = Image.open(os.path.join(self.root, 'Images', self.n_images[index])).convert('RGB')
        label = Image.open(self.labels[index]).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
            label = self.transform(label)
        label = torch.squeeze(label)
        return image, label

    def __len__(self):
        return len(self.n_images)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
 
trainval_set = PASCAL_Dataset(root='/home/vedpalj/personal/AIP_assignment/PascalVOC/trainval', transform=transform)
test_set = PASCAL_Dataset(root='/home/vedpalj/personal/AIP_assignment/PascalVOC/test', transform=transform)

train_data = int(0.8 * len(trainval_set))
val_data = len(trainval_set) - train_data
train_set, val_set = random_split(trainval_set, [train_data, val_data])

batch_size = 16
num_epochs = 10
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, shuffle=False)

model = MobileNetV2Unet(num_classes=21) 
model.to(device)  
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
 
# Training loop...
print('\n start training...\n')
for epoch in range(num_epochs):
     
    model.train()
    train_loss = 0.0
     
    for inputs, labels in (train_loader):
        inputs, labels = inputs.to(device), torch.argmax(labels, dim=1).to(device).float()

        optimizer.zero_grad()
         
        outputs = model(inputs)
        outputs = torch.argmax(outputs, dim=1).float()
        loss = criterion(outputs, labels)
        loss.requires_grad=True
        
        loss.backward()
        optimizer.step()
        train_loss += loss.sum().item()
    train_loss /= len(train_loader.dataset)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
         
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), torch.argmax(labels, dim=1).to(device).float()
            
            outputs = model(inputs)
            outputs = torch.argmax(outputs, dim=1).float()
            loss = criterion(outputs, labels)
            loss.requires_grad=True
            val_loss += loss.sum().item()
        val_loss /= len(val_loader.dataset)

    # Print loss for epoch
    if epoch % 2 == 0:
        print("Epoch: {}, Train Loss: {:.4f}, Val Loss: {:.4f}".format(epoch, train_loss, val_loss))

torch.save(model.state_dict(), "model.pt")

# testing...
model.eval()
total_correct = 0
total_pixels = 0
total_iou = 0
pix_wa = 0


# Test the model on the custom test set and compute pixelwise accuracy and meanIOU
with torch.no_grad():
    for images, targets in test_loader:
        images = images.to(device)
        targets =  torch.argmax(targets, dim=1).to('cpu').numpy()
        targets = (targets > 0)
         
        outputs = model(images)
        predicted = torch.argmax(outputs, dim=1).to('cpu').numpy()
        predicted = (predicted > 0)
        correct = (predicted == targets).sum()

        total_correct += correct
        total_pixels += (targets >= 0).sum()
        pix_wa += (total_correct / total_pixels)
        intersection = np.logical_and(predicted, targets).sum()
        union = np.logical_or(predicted, targets).sum()
        iou = (intersection + 1e-6) / (union + 1e-6)
        total_iou += iou.sum()

avg_pixelwise_accuracy = pix_wa / len(test_set)
mean_iou = total_iou / len(test_set)

print('--'*40)

print(f"Pixelwise accuracy: {avg_pixelwise_accuracy:.4f}")
print(f"MeanIOU: {mean_iou:.4f}")

print('--'*40)


