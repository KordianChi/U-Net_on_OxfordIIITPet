from unet_dataset import CuteDataset
from unet_model import UNet
import os
import time

from torch.utils.data import DataLoader
from torch.utils.data import random_split

from torchvision.transforms import transforms
from torch import cuda
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam

from matplotlib import pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


device = 'cuda:0' if cuda.is_available() else 'cpu'

TRAIN_IMG_DIR = r'data\images'
TRAIN_MASK_DIR = r'data\trimaps'
BATCH_SIZE = 100
epochs = 5
learning_rate = 0.0002

images = os.listdir(TRAIN_IMG_DIR)
masks = os.listdir(TRAIN_MASK_DIR)


transform_img = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((128, 128)),
                                    transforms.Normalize(mean=[0.0, 0.0, 0.0],
                                                         std=[1.0, 1.0, 1.0]),
                                   ])



transform_mask = transforms.Compose([transforms.ToTensor(),
                                     transforms.Resize((128, 128))
                                    ])

full_data = CuteDataset(images, TRAIN_IMG_DIR, TRAIN_MASK_DIR,
                         transform_img=transform_img,
                         transform_mask=transform_mask)

train_data, val_data = random_split(full_data, [0.8, 0.2])

train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)



unet_network = UNet().to(device)
criterion = BCEWithLogitsLoss()
optimizer = Adam(unet_network.parameters(), lr=learning_rate)

train_loss_data = []
val_loss_data = []


for _ in range(epochs):
    
    train_epoch_loss = []
    val_epoch_loss = []
    
    start = time.time()
    
    for step, data in enumerate(train_dataloader):
        
        step_start = time.time()
        
        image, mask = data["image"].to(device), data["mask"].to(device)
        optimizer.zero_grad()
        
        outputs = unet_network(image)
        
        train_loss = criterion(outputs, mask)
        train_loss.backward()
        train_epoch_loss.append(train_loss.item())
        
        optimizer.step()
        
        
        
        step_end = time.time()
        
        
        print(step_end - step_start)
        
    for step, data in enumerate(val_dataloader):
        
        image, mask = data["image"].to(device), data["mask"].to(device)
        outputs = unet_network(image)
        val_loss = criterion(outputs, mask)
        val_epoch_loss.append(val_loss.item())
        
    train_loss_data.append(sum(train_epoch_loss) / len(train_epoch_loss))
    val_loss_data.append(sum(val_epoch_loss) / len(val_epoch_loss))
        
    end = time.time()
    print(end - start)
    
    plt.plot(list(range(epochs)), train_loss_data, label='Train loss')
    plt.plot(list(range(epochs)), val_loss_data, label='Validation loss')
    plt.legend(loc="upper right")
    plt.xticks(list(range(1, epochs)))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')