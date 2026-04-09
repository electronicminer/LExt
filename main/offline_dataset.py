import os
import random
import torch
import torchaudio
from torch.utils.data import Dataset
from modern_vad import SileroVADProcessor

class OfflineLExtDataset(Dataset):
    def __init__(self, split_dir, sample_rate=8000, enrollment_len_s=4.0, is_training=True):
        self.is_training = is_training

        self.mix_dir = os.path.join(split_dir, "mix")
        self.s1_dir = os.path.join(split_dir, "s1")
        self.s2_dir = os.path.join(split_dir, "s2")
        
        self.sr = sample_rate
        self.enroll_len_samples = int(enrollment_len_s * self.sr)
        
        self.glue_len_samples = int(0.032 * self.sr)
        self.glue_signal = torch.full((1, self.glue_len_samples), 5.0)
        
        self.vad_processor = SileroVADProcessor()

        self.filenames = [f for f in os.listdir(self.mix_dir) if f.endswith('.wav')]

        print("建立说话人音频索引")
        self.spk_to_files = {}
        for fname in self.filenames:
            # WSJ0 的文件命名规范：前三个字符是说话人ID
            spk1_id = fname[:3] 
            spk2_id = fname.split('_')[2][:3]
            
            if spk1_id not in self.spk_to_files:
                self.spk_to_files[spk1_id] = []
            if spk2_id not in self.spk_to_files:
                self.spk_to_files[spk2_id] = []
                
            self.spk_to_files[spk1_id].append(os.path.join(self.s1_dir, fname))
            self.spk_to_files[spk2_id].append(os.path.join(self.s2_dir, fname))
            
        print(f"索引建立完成,发现 {len(self.filenames)} 条混合语音，{len(self.spk_to_files)} 个独立说话人。")

    # 简单的语音活动检测 (SAD)，用于切除注册语音里的静音（后续我不再使用简单的基于能量的sad）
    def _remove_silence(self, waveform, threshold=0.005):
        """基于能量的 VAD，剔除 WSJ0 里的静音片段"""
        # 计算 20ms 窗口，10ms 步长的短时能量
        window_size = int(0.02 * self.sr)
        hop_size = int(0.01 * self.sr)
        
        if waveform.shape[1] < window_size:
            return waveform
            
        unfolded = waveform.unfold(1, window_size, hop_size)
        energy = unfolded.pow(2).mean(dim=-1).squeeze(0)
        
        # 找到能量大于最大能量 * 阈值的活跃帧
        max_energy = energy.max()
        active_frames = energy > (max_energy * threshold)
        
        # 将活跃帧映射回时间轴
        mask = torch.repeat_interleave(active_frames, hop_size)
        if len(mask) < waveform.shape[1]:
            mask = torch.cat([mask, torch.zeros(waveform.shape[1] - len(mask), dtype=torch.bool)], dim=0)
        else:
            mask = mask[:waveform.shape[1]]
            
        active_speech = waveform[:, mask]
        
        if active_speech.shape[1] < self.sr * 0.5: 
            return waveform
        return active_speech

    def _apply_sad_and_pad(self, enroll_waveform):
        # 在处理前，先调用 SAD 切除原声里的长静音
        # active_speech = self._remove_silence(enroll_waveform)
        active_speech = self.vad_processor.remove_silence(enroll_waveform, self.sr)
        
        current_len = active_speech.shape[1]
        target_len = self.enroll_len_samples 
        
        if current_len < target_len:
            pad_len = target_len - current_len
            # 不足 4s 时向左侧补零
            return torch.nn.functional.pad(active_speech, (pad_len, 0), "constant", 0)
        else:
            if self.is_training:
                # 训练时随机截取 4s
                start = random.randint(0, current_len - target_len)
            else:
                # 评估时永远取前 4s
                start = 0 
            return active_speech[:, start : start + target_len]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        # 定长
        mix_len_samples = int(4.0 * self.sr) 
        
        mix, _ = torchaudio.load(os.path.join(self.mix_dir, fname))
        s1, _ = torchaudio.load(os.path.join(self.s1_dir, fname))
        s2, _ = torchaudio.load(os.path.join(self.s2_dir, fname))
        
        # 随机选择目标说话人 
        is_s1_target = random.choice([True, False])
        target_sig = s1 if is_s1_target else s2
        
        # 4. 只在训练阶段截取 4.0s 的混合片段
        if self.is_training:
            if mix.shape[1] > mix_len_samples:
                start = random.randint(0, mix.shape[1] - mix_len_samples)
                mix = mix[:, start : start + mix_len_samples]
                target_sig = target_sig[:, start : start + mix_len_samples]
            elif mix.shape[1] < mix_len_samples:
                pad_len = mix_len_samples - mix.shape[1]
                mix = torch.nn.functional.pad(mix, (0, pad_len))
                target_sig = torch.nn.functional.pad(target_sig, (0, pad_len))

        # enroll
        target_spk_id = fname[:3] if is_s1_target else fname.split('_')[2][:3]
        available_enroll_files = [f for f in self.spk_to_files[target_spk_id] if fname not in f]
        
        enroll_file = random.choice(available_enroll_files) if available_enroll_files else os.path.join(self.s1_dir if is_s1_target else self.s2_dir, fname)
        enroll_sig, _ = torchaudio.load(enroll_file)
        
        # 预处理
        enroll_sig = self._apply_sad_and_pad(enroll_sig)
        
        # 归一化
        sigma_y = torch.std(mix, unbiased=False) + 1e-8
        sigma_e = torch.std(enroll_sig, unbiased=False) + 1e-8
        
        target_norm = target_sig / sigma_y
        mix_norm = mix / sigma_y
        enroll_norm = enroll_sig / sigma_e
        
        # 拼接 
        network_input = torch.cat([enroll_norm, self.glue_signal, mix_norm], dim=1)
        network_target = torch.cat([enroll_norm, self.glue_signal, target_norm], dim=1)
        
        if self.is_training:
            return network_input, network_target
        else:
            return network_input, network_target, sigma_y

# 测试代码
if __name__ == "__main__":
    dataset_path = "./wsj0-2mix/2speakers/wav8k/min/tr" 
    
    if os.path.exists(dataset_path):
        dataset = OfflineLExtDataset(dataset_path)
        net_in, net_target = dataset[0]
        
        print(f"最终给网络的输入 Tensor 形状: {net_in.shape}")
        print(f"网络的预测目标 Tensor 形状: {net_target.shape}")
    else:
        print(f"not found {dataset_path}")