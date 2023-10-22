
from time import time
from tqdm import tqdm
import numpy as np
from PIL import Image 

import torch
from torch.nn import Linear, CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18
from torchvision.transforms import transforms
from torchsummary import summary


# device...
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Data Transformer...
tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# Create Dataset...
TRAIN_ROOT = "Images/Train"
TEST_ROOT = "Images/Test"

train_ds = ImageFolder(TRAIN_ROOT, transform=tfm)
test_ds = ImageFolder(TEST_ROOT, transform=tfm)


# Length of Train and Test Datasets
LEN_TRAIN = len(train_ds)
LEN_TEST = len(test_ds)

print('\n #################  CLASSES...  #####################')
classes = [i for i in test_ds.class_to_idx]
print(classes)

# Data Loader
train_loader = DataLoader(train_ds, batch_size = 32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=32, shuffle = True)


# Model
model = resnet18(pretrained=True)

# Replace Output of Fully Connected Layer with Number of Labels for our Classification Problem
model.fc = Linear(in_features=512, out_features=5)
model = model.to(device)

# Loss and Optimizer
loss_fn = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

# training procedure...

start = time()
for epoch in range(15):
    
    tr_acc = 0
    test_acc = 0
    
    # Train
    model.train()
    
    with tqdm(train_loader, unit="batch") as tepoch:
        for xtrain, ytrain in tepoch:
            optimizer.zero_grad()
            
            xtrain = xtrain.to(device)
            train_prob = model(xtrain)
            train_prob = train_prob.cpu()
            
            loss = loss_fn(train_prob, ytrain)
            loss.backward()
            optimizer.step()
            
            # training ends
            
            train_pred = torch.max(train_prob, 1).indices
            tr_acc += int(torch.sum(train_pred == ytrain))
            
        ep_tr_acc = tr_acc / LEN_TRAIN
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        for xtest, ytest in test_loader:
            xtest = xtest.to(device)
            test_prob = model(xtest)
            test_prob = test_prob.cpu()
            
            test_pred = torch.max(test_prob,1).indices
            test_acc += int(torch.sum(test_pred == ytest))
            
        ep_test_acc = test_acc / LEN_TEST
    

    
    print('')
    print(f"Epoch: {epoch+1}, \nLoss: {loss:3f}\nTrain_acc: {ep_tr_acc:3f}, \tTest_acc: {ep_test_acc:3f}")
    print("--"*20)

end = time()
duration = (end - start) / 60
print(f'training time... :: {duration}min')

torch.save(model.state_dict(), 'state_dict_model.pt')

model = resnet18(pretrained=True)
model.fc = Linear(in_features=512, out_features=5)
model.load_state_dict(torch.load('state_dict_model.pt'))
model = model.to(device)
model.eval()

print(summary(model, (3, 224, 224)))


# testing...
path = 'test.JPEG'   # test img...  inclueded in folder

img = Image.open(path)
img.resize((224,224))
img_tensor = tfm(img)
img_tensor = img_tensor[np.newaxis, :]
img_tensor = img_tensor.to(device)
pred_prob = model(img_tensor)
pred = torch.max(pred_prob,1).indices
pred = pred.item()

print(f'prediceted class... {classes[pred][:-5]}')

print("=="*20)


