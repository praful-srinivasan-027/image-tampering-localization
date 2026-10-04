from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from model import UNet


class CASIA_Dataset(Dataset):

    def __init__(self, target_size=(384, 256)):
        self.image_dirX = Path(
            "/kaggle/input/datasets/prafulsrinivasan/casia/casia/Tp"
        )
        self.image_dirY = Path(
            "/kaggle/input/datasets/prafulsrinivasan/casia/casia/Gt"
        )
        self.target_size = target_size

        all_x = []
        for suffix in ["*.tif", "*.jpg", "*.png"]:
            all_x.extend(self.image_dirX.glob(suffix.lower()))
            all_x.extend(self.image_dirX.glob(suffix.upper()))

        self.pairs = []
        for x in all_x:
            y = self.image_dirY / f"{x.stem}_gt.png"
            if y.exists():
                self.pairs.append((x, y))

        self.pairs.sort()
        print(f"Kept {len(self.pairs)} items with existing masks.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, elementNumber):
        x, y = self.pairs[elementNumber]

        imgX = cv2.imread(str(x), 1)
        imgY = cv2.imread(str(y), 0)

        imgX = cv2.resize(imgX, self.target_size)
        imgY = cv2.resize(
            imgY,
            self.target_size,
            interpolation=cv2.INTER_NEAREST
        )

        imgX = cv2.cvtColor(imgX, cv2.COLOR_BGR2RGB)

        imgX = imgX.astype(np.float32) / 255.0
        imgX = torch.from_numpy(imgX).permute(2, 0, 1)

        imgY = torch.from_numpy(
            (imgY >= 128).astype(np.int64)
        )

        return imgX, imgY


def accuracy_fn(preds, targets):
    preds = torch.argmax(preds, dim=1)

    correct = (preds == targets).sum().float()
    acc = correct / targets.numel()

    return acc.item()


dataset = CASIA_Dataset()

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

use_pin = device.type == "cuda"

loader = DataLoader(
    dataset=dataset,
    batch_size=16,
    shuffle=True,
    num_workers=2,
    pin_memory=use_pin,
)

model = UNet().to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0001
)

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

    print(
        f"Epoch {epoch+1:02d}/{epochs:02d} - "
        f"Loss: {avg_loss:.4f} - "
        f"Acc: {avg_acc:.4f}"
    )


torch.save(
    model.state_dict(),
    "/kaggle/working/unet_casia.pth"
)