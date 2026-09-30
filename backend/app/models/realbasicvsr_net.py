import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlockNoBN(nn.Module):
    """Residual block without Batch Normalization.
    
    Used throughout RealBasicVSR generator and image cleaning modules.
    """

    def __init__(self, mid_channels: int = 64, res_scale: float = 1.0):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.relu(self.conv1(x))) * self.res_scale


class ResidualBlocksWithInputConv(nn.Module):
    """Residual blocks preceded by an input convolution and LeakyReLU activation."""

    def __init__(self, in_channels: int, out_channels: int = 64, num_blocks: int = 20):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Sequential(*[ResidualBlockNoBN(out_channels) for _ in range(num_blocks)]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)


class PixelShufflePack(nn.Module):
    """Pixel Shuffle upsampling block."""

    def __init__(self, in_channels: int, out_channels: int, scale_factor: int = 2, upsample_kernel: int = 3):
        super().__init__()
        self.scale_factor = scale_factor
        self.upsample_conv = nn.Conv2d(
            in_channels,
            out_channels * scale_factor * scale_factor,
            kernel_size=upsample_kernel,
            padding=(upsample_kernel - 1) // 2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.pixel_shuffle(self.upsample_conv(x), self.scale_factor)


class ConvModule(nn.Module):
    """Basic convolution module with optional activation."""

    def __init__(self, ic: int, oc: int, k: int = 7, s: int = 1, p: int = 3, act: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(ic, oc, kernel_size=k, stride=s, padding=p, bias=True)
        self.act = nn.ReLU(inplace=False) if act else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return self.act(x) if self.act is not None else x


class SPyNetBasicModule(nn.Module):
    """Basic submodule of SPyNet for optical flow estimation."""

    def __init__(self):
        super().__init__()
        self.basic_module = nn.Sequential(
            ConvModule(8, 32),
            ConvModule(32, 64),
            ConvModule(64, 32),
            ConvModule(32, 16),
            ConvModule(16, 2, act=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.basic_module(x)


class SPyNet(nn.Module):
    """SPyNet for coarse-to-fine optical flow estimation."""

    def __init__(self):
        super().__init__()
        self.basic_module = nn.ModuleList([SPyNetBasicModule() for _ in range(6)])
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def flow_warp(
        self,
        x: torch.Tensor,
        flow: torch.Tensor,
        interpolation: str = "bilinear",
        padding_mode: str = "zeros",
    ) -> torch.Tensor:
        """Warp an image or feature map using optical flow."""
        n, c, h, w = x.size()
        gy, gx = torch.meshgrid(
            torch.arange(h, device=x.device, dtype=x.dtype),
            torch.arange(w, device=x.device, dtype=x.dtype),
            indexing="ij",
        )
        grid = torch.stack((gx, gy), 2).unsqueeze(0).repeat(n, 1, 1, 1)
        gf = grid + flow.permute(0, 2, 3, 1)
        gxs = 2.0 * gf[..., 0] / max(w - 1, 1) - 1.0
        gys = 2.0 * gf[..., 1] / max(h - 1, 1) - 1.0
        return F.grid_sample(
            x,
            torch.stack((gxs, gys), dim=3),
            mode=interpolation,
            padding_mode=padding_mode,
            align_corners=True,
        )

    def forward(self, ref: torch.Tensor, supp: torch.Tensor) -> torch.Tensor:
        n, c, h, w = ref.size()
        ref = [self.preprocess(ref)]
        supp = [self.preprocess(supp)]
        for _ in range(5):
            ref.append(F.avg_pool2d(ref[-1], 2, 2, count_include_pad=False))
            supp.append(F.avg_pool2d(supp[-1], 2, 2, count_include_pad=False))
        ref, supp = ref[::-1], supp[::-1]
        flow = ref[0].new_zeros(n, 2, ref[0].size(2), ref[0].size(3))
        for level in range(len(ref)):
            flow_up = (
                flow
                if level == 0
                else F.interpolate(flow, scale_factor=2, mode="bilinear", align_corners=True) * 2.0
            )
            flow_res = self.basic_module[level](
                torch.cat([ref[level], self.flow_warp(supp[level], flow_up), flow_up], dim=1)
            )
            flow = flow_up + flow_res
        return flow


class RealBasicVSRNet(nn.Module):
    """RealBasicVSR network for Anime Video / Image Super-Resolution x4.
    
    Accepts tensor of shape [B, T, C, H, W] in range [0, 1].
    Outputs tensor of shape [B, T, C, H*4, W*4] in range [0, 1].
    """

    def __init__(self, mid_channels: int = 64, num_blocks: int = 20):
        super().__init__()
        self.mid_channels = mid_channels
        self.image_cleaning = nn.Sequential(
            ResidualBlocksWithInputConv(3, 64, 20),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1, bias=True),
        )
        self.spynet = SPyNet()
        self.backward_resblocks = ResidualBlocksWithInputConv(mid_channels + 3, mid_channels, num_blocks)
        self.forward_resblocks = ResidualBlocksWithInputConv(mid_channels + 3, mid_channels, num_blocks)
        self.fusion = nn.Conv2d(mid_channels * 2, mid_channels, kernel_size=1, stride=1)
        self.upsample1 = PixelShufflePack(mid_channels, mid_channels, scale_factor=2, upsample_kernel=3)
        self.upsample2 = PixelShufflePack(mid_channels, 64, scale_factor=2, upsample_kernel=3)
        self.conv_hr = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv_last = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
        self.img_upsample = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, lqs: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            lqs: Tensor of shape [B, T, C, H, W] normalized to [0, 1].
            
        Returns:
            Tensor of shape [B, T, C, H*4, W*4].
        """
        n, t, c, h, w = lqs.size()
        flat = lqs.reshape(-1, c, h, w)
        clean = (self.image_cleaning(flat) + flat).clamp(0.0, 1.0).reshape(n, t, c, h, w)

        # Estimate bidirectional optical flows
        ff = []
        fb = []
        for i in range(t - 1):
            ff.append(self.spynet(clean[:, i + 1], clean[:, i]))
            fb.append(self.spynet(clean[:, i], clean[:, i + 1]))

        # Backward propagation
        backward = []
        feat = lqs.new_zeros(n, self.mid_channels, h, w)
        for i in range(t - 1, -1, -1):
            if i < t - 1:
                feat = self.spynet.flow_warp(feat, fb[i])
            feat = feat + self.backward_resblocks(torch.cat([clean[:, i], feat], dim=1))
            backward.append(feat)
        backward = backward[::-1]

        # Forward propagation and output generation
        outs = []
        feat = lqs.new_zeros(n, self.mid_channels, h, w)
        for i in range(t):
            if i > 0:
                feat = self.spynet.flow_warp(feat, ff[i - 1])
            feat = feat + self.forward_resblocks(torch.cat([clean[:, i], feat], dim=1))
            out = self.fusion(torch.cat([backward[i], feat], dim=1))
            out = self.lrelu(self.upsample1(out))
            out = self.lrelu(self.upsample2(out))
            out = self.lrelu(self.conv_hr(out))
            out = self.conv_last(out) + self.img_upsample(lqs[:, i])
            outs.append(out)

        return torch.stack(outs, dim=1)
