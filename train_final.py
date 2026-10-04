from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from model import UNet


class CASIA_Dataset(Dataset):

    def __init__(self, target_size=(384, 256)):
        self.image_dirX = Path("Dataset/casia/Tp")
        self.image_dirY = Path("Dataset/casia/Gt")
        self.target_size = target_size
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
        imgX = cv2.resize(imgX, self.target_size)
        imgY = cv2.resize(
            imgY, self.target_size, interpolation=cv2.INTER_NEAREST
        )
        imgX = cv2.cvtColor(imgX, cv2.COLOR_BGR2RGB)
        imgX = imgX.astype(np.float32) / 255.0
        imgX = torch.from_numpy(imgX).permute(2, 0, 1)
        imgY = torch.from_numpy((imgY >= 128).astype(np.int64))
        return imgX, imgY


def accuracy_fn(preds, targets):
    preds = torch.argmax(preds, dim=1)
    correct = (preds == targets).sum().float()
    acc = correct / targets.numel()
    return acc.item()


dataset = CASIA_Dataset()
loader = DataLoader(dataset=dataset, batch_size=8, shuffle=True)

device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
model = UNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
lossFn = nn.CrossEntropyLoss()
epochs = 20

for epoch in range(epochs):
    model.train()
    total_loss = 0
    total_acc = 0
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = lossFn(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += accuracy_fn(outputs, masks)

    avg_loss = total_loss / len(loader)
    avg_acc = total_acc / len(loader)
    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Acc: {avg_acc:.4f}")

torch.save(model.state_dict(), "unet_casia.pth")