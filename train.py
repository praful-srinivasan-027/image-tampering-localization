from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2

class CASIA_Dataset(Dataset):
    def __init__(self):
        #This is RGB
        self.image_dirX = Path("Dataset/casia/Tp")
        #This is GrayScale
        self.image_dirY = Path("Dataset/casia/Gt")
        self.x = []
        for suffix in ["*.tif", "*.jpg", "*.png"]:
            self.x.extend(self.image_dirX.glob(suffix.lower()))
        self.x.sort()

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, elementNumber):
        x = self.x[elementNumber]
        y = self.image_dirY / f"{x.stem}_gt.png"
        imgX = cv2.imread(str(x), 1)
        imgY = cv2.imread(str(y), 0)
        if imgX is None:
            raise FileNotFoundError(f"Couldn't load the image {x}")
        if imgY is None:
            raise FileNotFoundError(f"Couldn't load the image {y}")
        imgX = cv2.cvtColor(imgX, cv2.COLOR_BGR2RGB)
        #Normalize
        imgX = imgX.astype(np.float32)/255.0
        imgY = imgY.astype(np.float32)/255.0
        return torch.from_numpy(imgX), torch.from_numpy(imgY)

dataset = CASIA_Dataset()
loader = DataLoader(
    dataset=dataset,
    batch_size=8,
    shuffle=True
    )