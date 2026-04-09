"""
DeepRawNet: Empowering Deepfake Audio Detection through Dynamic Enhancements
Paper: Alharbi et al. (2026), PeerJ Comput. Sci., DOI 10.7717/peerj-cs.3670

Architecture based on RawNet2 with three key innovations:
  1. PReLU activation (learnable negative slope) in residual blocks → replaces LeakyReLU
  2. Transpose Convolution in residual blocks → replaces standard Conv (addresses downsampling)
  3. LogSoftmax in the output layer → replaces Softmax (numerical stability)

Additional change from paper:
  - Increased negative slope of LeakyReLU in Fixed Sinc filters to 0.5

Overfitting fixes for small datasets (e.g. 500 samples):
  - Dropout added inside ResidualBlock (default 0.3)
  - Dropout added in GRU (default 0.3)
  - Dropout added before FC layer (default 0.5)
  - L2 regularization via weight_decay in optimizer (already present)
  - All dropout rates controllable via dropout_rate argument

Usage:
    model = DeepRawNet(num_classes=2, dropout_rate=0.3)
    out = model(waveform)   # waveform: (batch, 1, samples)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ---------------------------------------------------------------------------
# Sinc-based convolution (Fixed Sinc filters) — taken from RawNet2 baseline
# ---------------------------------------------------------------------------

class SincConv(nn.Module):
    """
    Sinc-based bandpass filter bank applied directly on raw waveforms.
    The negative slope of the internal LeakyReLU is set to 0.5 (paper §Methods).
    """

    @staticmethod
    def to_mel(hz: float) -> float:
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel: float) -> float:
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(
        self,
        out_channels: int = 128,
        kernel_size: int = 251,
        sample_rate: int = 16000,
        in_channels: int = 1,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = False,
        groups: int = 1,
        min_low_hz: float = 50.0,
        min_band_hz: float = 50.0,
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1  # must be odd

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        # Initialise filter-bank edges from a Mel scale
        low_hz = 30.0
        high_hz = sample_rate / 2 - (min_low_hz + min_band_hz)

        mel = np.linspace(self.to_mel(low_hz), self.to_mel(high_hz), out_channels + 1)
        hz = self.to_hz(mel)

        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))

        # Hamming window
        n_lin = torch.linspace(0, kernel_size / 2 - 1, steps=kernel_size // 2)
        self.register_buffer("window_", 0.54 - 0.46 * torch.cos(2 * np.pi * n_lin / kernel_size))
        n = (kernel_size - 1) / 2.0
        self.register_buffer("n_", 2 * np.pi * torch.arange(-n, 0).view(1, -1) / sample_rate)

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        self.n_ = self.n_.to(waveforms.device)
        self.window_ = self.window_.to(waveforms.device)

        low = self.min_low_hz + torch.abs(self.low_hz_)
        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_), self.min_low_hz, self.sample_rate / 2)
        band = (high - low)[:, 0]

        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)

        band_pass_left = (torch.sin(f_times_t_high) - torch.sin(f_times_t_low)) / (self.n_ / 2) * self.window_
        band_pass_center = 2 * band.view(-1, 1)
        band_pass_right = torch.flip(band_pass_left, dims=[1])

        band_pass = torch.cat([band_pass_left, band_pass_center, band_pass_right], dim=1)
        band_pass = band_pass / (2 * band[:, None])

        filters = band_pass.view(self.out_channels, 1, self.kernel_size)
        return F.conv1d(waveforms, filters, stride=self.stride, padding=self.padding, dilation=self.dilation)


class SincConvBlock(nn.Module):
    """
    First stage: Sinc filter bank → MaxPool → BN → LeakyReLU(0.5)
    Paper: 'negative slope in the Fixed Sinc filters is increased to 0.5'
    """

    def __init__(self, out_channels: int = 128, kernel_size: int = 251, sample_rate: int = 16000):
        super().__init__()
        self.sinc = SincConv(out_channels=out_channels, kernel_size=kernel_size,
                             sample_rate=sample_rate, padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(out_channels)
        self.pool = nn.MaxPool1d(kernel_size=3)
        # Paper: increased negative slope to 0.5 in Sinc filters
        self.activation = nn.LeakyReLU(negative_slope=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.sinc(x)           # (B, C, T)
        x = self.pool(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


# ---------------------------------------------------------------------------
# Filter-wise Feature Map Scaling (FMS) — from RawNet2
# ---------------------------------------------------------------------------

class FMS(nn.Module):
    """
    Filter-wise Feature Map Scaling — a lightweight channel attention mechanism.
    Scales each channel independently using a learned sigmoid weight.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.fc = nn.Linear(channels, channels)
        self.sig = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        scale = x.mean(dim=2)           # (B, C) global average pooling
        scale = self.sig(self.fc(scale))  # (B, C) learned scaling
        x = x * scale.unsqueeze(2)     # (B, C, T) channel-wise multiply
        return x


# ---------------------------------------------------------------------------
# DeepRawNet Residual Block
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """
    Enhanced residual block with two key changes over RawNet2:
      1. PReLU (learnable negative slope) replaces LeakyReLU  [paper §Methods]
      2. Transpose Convolution replaces standard Conv to address downsampling
         and preserve fine-grained temporal information         [paper §Methods]

    Structure:
        BN → PReLU → TransposeConv → BN → PReLU → TransposeConv
        ⊕ skip → MaxPool → FMS
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dropout_rate: float = 0.3):
        super().__init__()
        padding = kernel_size // 2

        # Main path
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.prelu1 = nn.PReLU()                          # Innovation 1: PReLU
        # Innovation 2: Transpose convolution
        self.tconv1 = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, padding=padding)

        self.bn2 = nn.BatchNorm1d(out_channels)
        self.prelu2 = nn.PReLU()                          # Innovation 1: PReLU
        self.tconv2 = nn.ConvTranspose1d(out_channels, out_channels, kernel_size, padding=padding)

        # Overfitting fix: dropout after each conv in residual block
        self.dropout = nn.Dropout(p=dropout_rate)

        # Downsampling and attention
        self.pool = nn.MaxPool1d(kernel_size=3)
        self.fms = FMS(out_channels)

        # 1×1 conv to match channel dimensions on skip path when needed
        self.skip_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Main path
        out = self.bn1(x)
        out = self.prelu1(out)
        out = self.tconv1(out)
        out = self.dropout(out)          # Overfitting fix: dropout after first conv

        out = self.bn2(out)
        out = self.prelu2(out)
        out = self.tconv2(out)
        out = self.dropout(out)          # Overfitting fix: dropout after second conv

        # Align temporal length of skip with main path (transpose conv may add 1 sample)
        skip = self.skip_conv(x)
        min_len = min(out.size(2), skip.size(2))
        out = out[:, :, :min_len]
        skip = skip[:, :, :min_len]

        # Residual addition
        out = out + skip

        # Downsampling + FMS
        out = self.pool(out)
        out = self.fms(out)
        return out


# ---------------------------------------------------------------------------
# DeepRawNet — Full Model
# ---------------------------------------------------------------------------

class DeepRawNet(nn.Module):
    """
    DeepRawNet: deepfake audio detection model.

    Architecture (matching paper Fig. 2 & Fig. 3):
        Raw Audio
            ↓
        SincConvBlock (128 filters, kernel=251, LeakyReLU slope=0.5)
            ↓
        2 × ResidualBlock (128 channels, TransposeConv + PReLU + FMS)
            ↓
        4 × ResidualBlock (256 channels, TransposeConv + PReLU + FMS)
            ↓
        GRU (1024 hidden units, batch_first=True)
            ↓
        FC (1024 → num_classes)
            ↓
        LogSoftmax                    [Innovation 3]

    Args:
        num_classes (int): Number of output classes (default 2: bonafide / spoof).
        sample_rate (int): Audio sample rate in Hz (default 16000).
        dropout_rate (float): Dropout probability for regularization (default 0.3).
                              Increase to 0.4-0.5 if still overfitting on small datasets.
    """

    def __init__(self, num_classes: int = 2, sample_rate: int = 16000, dropout_rate: float = 0.5):
        super().__init__()

        # Stage 1: Sinc filter bank with increased LeakyReLU slope
        self.sinc_block = SincConvBlock(out_channels=128, kernel_size=251, sample_rate=sample_rate)

        # Stage 2: 2 residual blocks at 128 channels
        self.res_blocks_128 = nn.Sequential(
            ResidualBlock(in_channels=128, out_channels=128, dropout_rate=dropout_rate),
            ResidualBlock(in_channels=128, out_channels=128, dropout_rate=dropout_rate),
        )

        # Stage 3: 4 residual blocks at 256 channels
        self.res_block_up = ResidualBlock(in_channels=128, out_channels=256, dropout_rate=dropout_rate)
        self.res_blocks_256 = nn.Sequential(
            ResidualBlock(in_channels=256, out_channels=256, dropout_rate=dropout_rate),
            ResidualBlock(in_channels=256, out_channels=256, dropout_rate=dropout_rate),
            ResidualBlock(in_channels=256, out_channels=256, dropout_rate=dropout_rate),
        )

        # Stage 4: GRU with dropout for regularization
        self.gru = nn.GRU(
            input_size=256,
            hidden_size=1024,
            num_layers=1,
            batch_first=True,
            dropout=0.0,   # dropout only applies when num_layers > 1 in GRU
        )
        # Overfitting fix: explicit dropout after GRU output
        self.gru_dropout = nn.Dropout(p=dropout_rate)

        # Stage 5: FC with dropout + LogSoftmax
        # Overfitting fix: dropout before FC to prevent co-adaptation
        self.fc_dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(1024, num_classes)
        # Innovation 3: LogSoftmax for numerical stability
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Raw audio waveform — shape (batch, 1, num_samples)
        Returns:
            log_probs: Log-probability scores — shape (batch, num_classes)
        """
        # Sinc filter block
        x = self.sinc_block(x)          # (B, 128, T1)

        # Residual blocks — 128 channels
        x = self.res_blocks_128(x)      # (B, 128, T2)

        # Transition + residual blocks — 256 channels
        x = self.res_block_up(x)        # (B, 256, T3)
        x = self.res_blocks_256(x)      # (B, 256, T4)

        # GRU expects (B, T, C)
        x = x.permute(0, 2, 1)         # (B, T4, 256)
        x, _ = self.gru(x)             # (B, T4, 1024)
        x = x[:, -1, :]               # last time step → (B, 1024)
        x = self.gru_dropout(x)        # Overfitting fix: dropout after GRU

        # Classification head
        x = self.fc_dropout(x)         # Overfitting fix: dropout before FC
        x = self.fc(x)                 # (B, num_classes)
        x = self.log_softmax(x)        # (B, num_classes)
        return x


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def get_optimizer(model: nn.Module, lr: float = 1e-4, weight_decay: float = 1e-4):
    """
    ADAM optimizer with the hyperparameters reported in the paper.
    Paper: lr=0.0001, weight_decay=0.0001
    """
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def get_loss():
    """
    Cross-entropy loss used during training.
    Note: nn.NLLLoss pairs with LogSoftmax output (equivalent to CrossEntropyLoss).
    """
    return nn.NLLLoss()


# ---------------------------------------------------------------------------
# Quick sanity-check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}\n")

    # dropout_rate=0.3 recommended for ~500 samples; increase to 0.4-0.5 if still overfitting
    model = DeepRawNet(num_classes=2, sample_rate=16000, dropout_rate=0.3).to(device)

    # Simulate a batch of 4 audio samples, each 4 seconds at 16 kHz
    dummy = torch.randn(4, 1, 64_000).to(device)

    with torch.no_grad():
        out = model(dummy)

    print("DeepRawNet architecture:\n")
    print(model)
    print(f"\nInput shape  : {dummy.shape}")
    print(f"Output shape : {out.shape}")
    print(f"Output (log-probs):\n{out}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters     : {total_params:,}")
    print(f"Trainable parameters : {trainable_params:,}")