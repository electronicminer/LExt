import os
import torch
import torchaudio
import torchaudio.transforms as T
from model_and_loss import get_tfgridnet_v2_model


class SileroVADProcessor:
    def __init__(self):
        print("load vad...")
        self.model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True
        )
        self.get_speech_timestamps = utils[0]
        self.collect_chunks = utils[4]
        print("vad loaded")

    def remove_silence(self, waveform, sr=8000):
        wav_1d = waveform.squeeze(0)
        # 过滤小杂音
        speech_timestamps = self.get_speech_timestamps(
            wav_1d, self.model, sampling_rate=sr,
            threshold=0.5, min_speech_duration_ms=250, min_silence_duration_ms=100
        )
        if not speech_timestamps:
            return waveform
            
        active_speech_1d = self.collect_chunks(speech_timestamps, wav_1d)
        if len(active_speech_1d) < sr * 0.5:
            return waveform
        return active_speech_1d.unsqueeze(0)

# 前处理
def process_enrollment(enroll_sig, vad_processor, sr=8000, enroll_len_s=4.0):
    # 切静音
    active_speech = vad_processor.remove_silence(enroll_sig, sr)
    target_len = int(enroll_len_s * sr)
    current_len = active_speech.shape[1]
    
    if current_len < target_len:
        pad_len = target_len - current_len
        return torch.nn.functional.pad(active_speech, (pad_len, 0), "constant", 0)
    else:
        # 取前4s
        return active_speech[:, :target_len]

# 推理
def separate_custom_audio(mix_path, enroll_path, weight_path, output_path=r"demo\extracted_target.wav"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device:', device)
    
    vad_processor = SileroVADProcessor()

    model = get_tfgridnet_v2_model().to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
    model.eval()
    print("load ckpt done")

    # resample to 8k
    target_sr = 8000
    mix, mix_sr = torchaudio.load(mix_path)
    enroll, enroll_sr = torchaudio.load(enroll_path)
    
    if mix_sr != target_sr:
        mix = T.Resample(mix_sr, target_sr)(mix)
    if enroll_sr != target_sr:
        enroll = T.Resample(enroll_sr, target_sr)(enroll)
        
    # to mono
    if mix.shape[0] > 1: mix = mix[0:1, :]
    if enroll.shape[0] > 1: enroll = enroll[0:1, :]

    # 切静音
    print("vad processing...")
    enroll_processed = process_enrollment(enroll, vad_processor, target_sr)

    # 归一化
    sigma_y = torch.std(mix, unbiased=False) + 1e-8
    sigma_e = torch.std(enroll_processed, unbiased=False) + 1e-8
    mix_norm = mix / sigma_y
    enroll_norm = enroll_processed / sigma_e

    # glue
    glue_samples = int(0.032 * target_sr)
    glue_signal = torch.full((1, glue_samples), 5.0)
    
    network_input = torch.cat([enroll_norm, glue_signal, mix_norm], dim=1).unsqueeze(0).to(device)

    print("forwarding...")
    with torch.no_grad():
        est_target = model(network_input)

    # 还原音量
    discard_len = int((4.0 + 0.032) * target_sr)
    est_audio = est_target.squeeze()[discard_len:] * sigma_y.to(device)

    max_val = torch.max(torch.abs(est_audio))
    if max_val > 0:
        # 缩放一下要不然破音
        est_audio = (est_audio / max_val) * 0.95

    torchaudio.save(output_path, est_audio.cpu().unsqueeze(0), target_sr)
    print(f"done: {output_path}")

if __name__ == "__main__":
    # configs
    CUSTOM_MIX = r"demo\mix.wav"          
    CUSTOM_ENROLL = r"demo\enrollment.wav"   
    MODEL_WEIGHT = r"main\lext_tfgridnet_epoch_22.pth" #
    
    if os.path.exists(CUSTOM_MIX) and os.path.exists(CUSTOM_ENROLL) and os.path.exists(MODEL_WEIGHT):
        separate_custom_audio(CUSTOM_MIX, CUSTOM_ENROLL, MODEL_WEIGHT)
    else:
        print("file not found")