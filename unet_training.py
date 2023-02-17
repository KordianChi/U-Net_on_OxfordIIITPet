from unet_dataset import CuteDataset
from unet_model import UNet
import os
import time

from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torch import cuda
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


device = 'cuda:0' if cuda.is_available() else 'cpu'

TRAIN_IMG_DIR = r'data\images'
TRAIN_MASK_DIR = r'data\trimaps'
BATCH_SIZE = 100
epochs = 1
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

train_data = CuteDataset(images, TRAIN_IMG_DIR, TRAIN_MASK_DIR,
                         transform_img=transform_img,
                         transform_mask=transform_mask)

train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)



unet_network = UNet().to(device)
criterion = BCEWithLogitsLoss()
optimizer = Adam(unet_network.parameters(), lr=learning_rate)

for _ in range(epochs):
    
    start = time.time()
    
    for step, data in enumerate(train_dataloader):
        
        step_start = time.time()
        
        image, mask = data["image"].to(device), data["mask"].to(device)
        optimizer.zero_grad()
        
        outputs = unet_network(image)
        
        outputs =outputs.squeeze(1)
        
        loss = criterion(outputs, mask)
        loss.backward()
        optimizer.step()
        
        step_end = time.time()
        
        print(step_end - step_start)
        
    end = time.time()
    print(end - start)