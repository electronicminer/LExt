import torch

class SileroVADProcessor:
    def __init__(self):
        print("load silero vad...")

        self.model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True
        )
        
        # utils
        self.get_speech_timestamps = utils[0]
        self.collect_chunks = utils[4]
        print("vad done")

    def remove_silence(self, waveform, sr=8000):
        # to 1d
        wav_1d = waveform.squeeze(0)

        speech_timestamps = self.get_speech_timestamps(
            wav_1d,
            self.model,
            sampling_rate=sr,
            threshold=0.5,                 
            min_speech_duration_ms=250,     
            min_silence_duration_ms=100     
        )

        # fail safe
        if not speech_timestamps:
            return waveform

        # 拼接
        active_speech_1d = self.collect_chunks(speech_timestamps, wav_1d)

        # 太短就不搞了
        if len(active_speech_1d) < sr * 0.5:
            return waveform

        # to unsqueeze
        return active_speech_1d.unsqueeze(0)