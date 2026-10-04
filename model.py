import torch
import torch.nn as nn
import cv2
import numpy as np

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.convBlock1 = self.provideConv2d(3, 64)
        self.convBlock2 = self.provideConv2d(64, 128)
        self.convBlock3 = self.provideConv2d(128, 256)
        self.convBlock4 = self.provideConv2d(256, 512)
        self.convBlockBottom = nn.Sequential(
            nn.Conv2d(
                in_channels=512,
                out_channels=1024,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=1024,
                out_channels=1024,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU()
        )
        self.maxpool = nn.MaxPool2d(
                        kernel_size=2,
                        stride=2,
                    )          
        self.upsampler=self.upsample()
        self.decoderBlock1=self.Decoder(1024, 512)
        self.decoderBlock2=self.Decoder(512, 256)
        self.decoderBlock3=self.Decoder(256, 128)
        self.decoderBlock4=self.Decoder(128, 64)
        self.finalConv = nn.Conv2d(
            in_channels=64,
            out_channels=2,
            kernel_size=1,
            stride=1,
            padding=0
        )
    def provideConv2d(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU()
        )
    def upsample(self):
        return nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    def Decoder(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels+out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU()
        )
    
    def forward(self, x):
        out1=self.convBlock1(x)
        out1_=self.maxpool(out1)
        out2=self.convBlock2(out1_)
        out2_=self.maxpool(out2)
        out3=self.convBlock3(out2_)
        out3_=self.maxpool(out3)
        out4=self.convBlock4(out3_)
        out4_=self.maxpool(out4)
        out5=self.convBlockBottom(out4_)
        out6=self.upsampler(out5)
        InputVector = torch.cat([out4, out6], dim=1)
        out7=self.decoderBlock1(InputVector)
        out8=self.upsampler(out7)
        InputVector = torch.cat([out3,out8], dim=1)
        out9=self.decoderBlock2(InputVector)
        out10=self.upsampler(out9)
        InputVector = torch.cat([out2, out10], dim=1)
        out11=self.decoderBlock3(InputVector)
        out12=self.upsampler(out11)
        InputVector = torch.cat([out1, out12], dim=1)
        out13=self.decoderBlock4(InputVector)
        out14 = self.finalConv(out13)
        return out14

imgX = cv2.imread("/Users/mac/Coding/Artificial Intelligence/ImageTampering/Dataset/casia/Tp/Tp_S_NRN_S_N_txt00070_txt00070_11315.jpg", 1)
imgX = cv2.cvtColor(imgX, cv2.COLOR_BGR2RGB)
print(imgX.shape)
model=UNet()
imgY = model.forward(torch.from_numpy(imgX.astype(np.float32)/255.0).permute(2, 0, 1).unsqueeze(0))
imgY = torch.argmax(imgY, dim=1)
print(imgY.shape)
cv2.imshow("Final", imgY.squeeze(0).detach().cpu().numpy().astype('uint8')*255)
# cv2.imshow("mat", imgX)
cv2.waitKey(0)
cv2.destroyAllWindows()