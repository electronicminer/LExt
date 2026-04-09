import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

from offline_dataset import OfflineLExtDataset
from model_and_loss import LExtLoss, get_tfgridnet_v2_model

def train_lext():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device:', device)
    
    batch_size = 4      
    num_epochs = 50
    learning_rate = 1e-3
    
    train_dir = r"./wsj0-2mix/2speakers/wav8k/min/tr"
    
    print("load data...")
    train_dataset = OfflineLExtDataset(split_dir=train_dir, sample_rate=8000, enrollment_len_s=4.0)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    

    model = get_tfgridnet_v2_model()
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    

    criterion = LExtLoss(sample_rate=8000, enrollment_len_s=4.0, glue_len_ms=32.0).to(device)
    

    print("start training...")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, (network_input, network_target) in enumerate(progress_bar):
            network_input = network_input.to(device)
            network_target = network_target.to(device)
            
            optimizer.zero_grad()
            
            est_target = model(network_input)
            
            loss = criterion(est_target, network_target)

            loss.backward()


            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        avg_loss = epoch_loss / len(train_loader)
        print(f"ep {epoch+1} loss: {avg_loss:.4f}\n")
        
        if (epoch + 1) % 5 == 0:
            save_path = f"lext_tfgridnet_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"saved {save_path}")

if __name__ == "__main__":
    train_lext()