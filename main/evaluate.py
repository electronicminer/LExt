import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio

# load
from offline_dataset import OfflineLExtDataset
from model_and_loss import get_tfgridnet_v2_model

def evaluate_lext():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device:', device)

    test_dir = r"./wsj0-2mix/2speakers/wav8k/min/tt"

    model_weight_path = r"main/lext_tfgridnet_epoch_22.pth" 
    
    sample_rate = 8000
    enrollment_len_s = 4.0
    glue_len_ms = 32.0
    
    # 算一下要扔掉多长 (e+g)
    discard_len = int((enrollment_len_s + glue_len_ms/1000.0) * sample_rate)

    # 准备数据集
    test_dataset = OfflineLExtDataset(
        split_dir=test_dir, 
        sample_rate=sample_rate, 
        enrollment_len_s=enrollment_len_s,
        is_training=False 
    )
    
    # audio len is variable, so bs=1
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4 )
    
    model = get_tfgridnet_v2_model().to(device)
    
    if os.path.exists(model_weight_path):
        model.load_state_dict(torch.load(model_weight_path, map_location=device))
        print(f"loaded {model_weight_path}")
    else:
        print(f"not found {model_weight_path}")

    model.eval()
    si_sdr_calc = ScaleInvariantSignalDistortionRatio().to(device)
    

    total_si_sdr_i = 0.0
    total_mixtures = len(test_loader)
    
    print(f"evaluating {total_mixtures} files...")
    
    with torch.no_grad():
        for batch_idx, (network_input, network_target, sigma_y) in enumerate(tqdm(test_loader)):
            network_input = network_input.to(device)
            network_target = network_target.to(device)
            sigma_y = sigma_y.to(device).unsqueeze(1)

            est_target = model(network_input)
            
            # to [B, T]
            if est_target.ndim == 3:
                est_target = est_target.squeeze(1)
            if network_target.ndim == 3:
                network_target = network_target.squeeze(1)
            if network_input.ndim == 3:
                network_input = network_input.squeeze(1)
                
            # 去掉前面
            est_target_clipped = est_target[:, discard_len:]
            clean_target_clipped = network_target[:, discard_len:]
            mixture_clipped = network_input[:, discard_len:] # mix

            # 乘回原来的标差
            est_target_restored = est_target_clipped * sigma_y
            clean_target_restored = clean_target_clipped * sigma_y
            mixture_restored = mixture_clipped * sigma_y
            
            # 去算 si-sdr
            sdr_processed = si_sdr_calc(est_target_restored, clean_target_restored)
            
            # 计算 base 的 si-sdr
            sdr_unprocessed = si_sdr_calc(mixture_restored, clean_target_restored)
            
            # improvement
            sdr_improvement = sdr_processed - sdr_unprocessed
            
            total_si_sdr_i += sdr_improvement.item()

    avg_si_sdr_i = total_si_sdr_i / total_mixtures
    print("\n" + "="*40)
    print(f"result :")
    print(f"SI-SDRi : {avg_si_sdr_i:.2f} dB")
    print("="*40)
    


if __name__ == "__main__":
    evaluate_lext()