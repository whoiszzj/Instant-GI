from torch import nn
from generalizable_model.convnext_unet import encoder
from generalizable_model.convnext_unet.decoder import UnetDecoder

ENCODERS = [
    'convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large', 'convnext_xlarge',
]
ENCODER_DIMS = {
    'convnext_tiny': [96, 192, 384, 768],
    'convnext_small': [96, 192, 384, 768],
    'convnext_base': [128, 256, 512, 1024],
    'convnext_large': [192, 384, 768, 1536],
    'convnext_xlarge': [256, 512, 1024, 2048],
}


class InConv(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.in_conv(x)


class ConvNeXtUnet(nn.Module):
    def __init__(
            self, out_channels, encoder_name,
            pretrained=False, in_22k=False, in_channels=3, bilinear=False, **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        self.dims = ENCODER_DIMS[encoder_name]
        self.in_conv = InConv(self.in_channels, self.dims[0] // 2)

        try:
            encoder_model = getattr(encoder, encoder_name)
        except AttributeError:
            raise AttributeError(
                f"Wrong encoder name '{encoder_name}'. Available options are: {ENCODERS}"
            )

        self.convnext_encoder = encoder_model(pretrained, in_22k, **kwargs)

        self.unet_decoder = UnetDecoder(
            out_channels=self.out_channels, dims=self.dims,
            in_channels=self.dims[0] // 2, bilinear=self.bilinear
        )

    def forward(self, x):
        x = self.in_conv(x)
        x, features = self.convnext_encoder(x)
        x = self.unet_decoder(x, features)
        return x


if __name__ == "__main__":
    import torch

    model = ConvNeXtUnet(
        out_channels=64, encoder_name='convnext_base',
        activation='sigmoid', pretrained=False, in_22k=False
    )
    x = torch.randn(1, 3, 288, 364)
    y = model(x)
    print(y.shape)
    print(y)
    print(model)
