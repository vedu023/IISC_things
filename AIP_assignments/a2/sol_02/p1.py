import os, torch, warnings
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models.segmentation import fcn_resnet50

warnings.filterwarnings("ignore")

class TestDataset(Dataset):
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
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
 
test_set = TestDataset(root='/home/vedpalj/personal/AIP_assignment/PascalVOC/test', transform=transform) 
test_loader = DataLoader(test_set, shuffle=False)

model = fcn_resnet50(pretrained=True) 
model.eval() 
model.to(device)

# Define variables to keep track of pixelwise accuracy and meanIOU
total_correct = 0
total_pixels = 0
total_iou = 0
pix_wa = 0


# Test the model on the custom test set and compute pixelwise accuracy and meanIOU
with torch.no_grad():
    for images, targets in test_loader:
        images = images.to(device)
        targets = targets.to('cpu').numpy() 
        targets = (targets > 0)
        
        outputs = model(images)['out'][0]
        predicted = torch.argmax (outputs.squeeze(), dim=0).cpu().numpy()
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