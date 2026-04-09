import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio

from models.local.TFgridnet import GridNetV2Block

class TFGridNetV2Wrapper(nn.Module):
    def __init__(self, 
                 emb_dim=128, 
                 num_blocks=6, 
                 hidden_channels=256,
                 n_fft=128,    # 16ms @ 8kHz
                 hop_length=64 # 8ms @ 8kHz
                 ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        n_freqs = n_fft // 2 + 1
        
        # input conv [B, F, T, 2] -> [B, D, T, F]
        self.input_conv = nn.Conv2d(2, emb_dim, kernel_size=1)
        
        # blocks
        self.blocks = nn.ModuleList([
            GridNetV2Block(
                emb_dim=emb_dim,
                emb_ks=1, # ks
                emb_hs=1, # stride
                n_freqs=n_freqs,
                hidden_channels=hidden_channels,
                n_head=4
            ) for _ in range(num_blocks)
        ])
        
        # output conv
        self.output_conv = nn.Conv2d(emb_dim, 2, kernel_size=1)

    def forward(self, wave):
        if wave.ndim == 3:
            wave = wave.squeeze(1)
        
        # stft
        window = torch.sqrt(torch.hann_window(self.n_fft)).to(wave.device)
        spec = torch.stft(wave, n_fft=self.n_fft, hop_length=self.hop_length, 
                          window=window, return_complex=True)
        # spec: [Batch, F, T]
        
        # ri [B, 2, F, T]
        spec_ri = torch.stack([spec.real, spec.imag], dim=1)
        
        # forward
        x = self.input_conv(spec_ri) # [B, D, F, T]
        x = x.transpose(2, 3)        # [B, D, T, F]
        
        for block in self.blocks:
            x = block(x)
            
        x = x.transpose(2, 3)        # [B, D, F, T]
        out_ri = self.output_conv(x) # [B, 2, F, T]
        
        # istft
        out_spec = torch.complex(out_ri[:, 0, :, :], out_ri[:, 1, :, :])
        out_wave = torch.istft(out_spec, n_fft=self.n_fft, hop_length=self.hop_length, 
                               window=window, length=wave.shape[-1])
        
        return out_wave.unsqueeze(1) # [Batch, Time]

# Loss
class LExtLoss(nn.Module):
    def __init__(self, sample_rate=8000, enrollment_len_s=4.0, glue_len_ms=32.0):
        super().__init__()
        self.sr = sample_rate
        self.discard_len = int((enrollment_len_s + glue_len_ms/1000.0) * self.sr)
        self.si_sdr = ScaleInvariantSignalDistortionRatio()

    def forward(self, est_target, true_target):
        # to [B, T]
        if est_target.ndim == 3:
            est_target = est_target.squeeze(1)
        if true_target.ndim == 3:
            true_target = true_target.squeeze(1)
            
        # 丢弃 E+G 去算loss
        loss = -self.si_sdr(est_target[:, self.discard_len:], true_target[:, self.discard_len:])
        return loss.mean()

def get_tfgridnet_v2_model():
    # config
    return TFGridNetV2Wrapper(
        emb_dim=128,        # D=128
        num_blocks=6,       # B=6
        hidden_channels=256 # H=256
    )

if __name__ == "__main__":
    # test
    model = get_tfgridnet_v2_model()
    test_input = torch.randn(1, 64256)
    output = model(test_input)
    print('in:', test_input.shape, 'out:', output.shape)