import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels = 1, out_channels = 1):
        super().__init__()

        # Encoder Blocks
        self.encoder_1 = self.conv_block(in_channels, 64)
        self.encoder_2 = self.conv_block(64, 128)
        self.encoder_3 = self.conv_block(128, 256)
        self.encoder_4 = self.conv_block(256, 512)

        self.bottleneck = self.conv_block(512, 1024)
        
        self.pool = nn.MaxPool2d(kernel_size=2)

        # Decoder Blocks
        self.upconv_4  = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.decoder_4 = self.conv_block(1024, 512)

        self.upconv_3  = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.decoder_3 = self.conv_block(512, 256)
        
        self.upconv_2  = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.decoder_2 = self.conv_block(256, 128)
        
        self.upconv_1  = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.decoder_1 = self.conv_block(128, 64)

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, n_input_channels: int, n_output_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(n_input_channels, n_output_channels, 3, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(n_output_channels, n_output_channels, 3, padding = 1),
            nn.ReLU(inplace = True),
        )
    
    def forward(self, x):
        # Encoding
        e1 = self.encoder_1(x)
        e2 = self.encoder_2(self.pool(e1))
        e3 = self.encoder_3(self.pool(e2))
        e4 = self.encoder_4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        # Decoding
        d4 = self.upconv_4(b)
        d4 = self.decoder_4(torch.cat([d4, e4], dim=1))

        d3 = self.upconv_3(d4)
        d3 = self.decoder_3(torch.cat([d3, e3], dim=1))

        d2 = self.upconv_2(d3)
        d2 = self.decoder_2(torch.cat([d2, e2], dim=1))

        d1 = self.upconv_1(d2)
        d1 = self.decoder_1(torch.cat([d1, e1], dim=1))

        out = self.final(d1)

        return out

        

        