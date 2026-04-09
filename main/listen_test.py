import os
import torch
import torchaudio
from offline_dataset import OfflineLExtDataset
from model_and_loss import get_tfgridnet_v2_model

def generate_demo():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device:', device)

    test_dir = r"wsj0-2mix/2speakers/wav8k/min/tt" 
    model_weight_path =r"main\lext_tfgridnet_epoch_22.pth"     

    dataset = OfflineLExtDataset(split_dir=test_dir, sample_rate=8000, enrollment_len_s=4.0, is_training=False)
    
    model = get_tfgridnet_v2_model().to(device)
    model.load_state_dict(torch.load(model_weight_path, map_location=device, weights_only=True))
    model.eval() 

    network_input, network_target, sigma_y = dataset[200]
    
    network_input = network_input.unsqueeze(0).to(device)
    sigma_y = sigma_y.to(device)

    with torch.no_grad():
        est_target = model(network_input)

    # 剔除前缀
    discard_len = int((4.0 + 0.032) * 8000)
    
    # 乘标差还原
    mix_audio = network_input.squeeze()[discard_len:] * sigma_y
    clean_audio = network_target.squeeze().to(device)[discard_len:] * sigma_y
    est_audio = est_target.squeeze()[discard_len:] * sigma_y

    max_val = torch.max(torch.abs(est_audio))
    if max_val > 0:
        # 缩放
        est_audio = (est_audio / max_val) * 0.95

    torchaudio.save("01_mixture.wav", mix_audio.cpu().unsqueeze(0), 8000)
    torchaudio.save("02_clean_target.wav", clean_audio.cpu().unsqueeze(0), 8000)
    torchaudio.save("03_model_estimated.wav", est_audio.cpu().unsqueeze(0), 8000)


if __name__ == "__main__":
    generate_demo()