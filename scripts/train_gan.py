import sys
import os
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pandas as pd
import numpy as np
import gc
from tqdm import tqdm

class DBLPDataset(Dataset):
    def __init__(self, data_path):
        print(f"[INFO] Loading dataset from {data_path}...")
        
        chunks = []
        try:
            for chunk in pd.read_csv(data_path, sep='\t', header=None, names=['h', 'r', 't'], dtype=np.int32, chunksize=1000000):
                chunks.append(chunk)
            
            df = pd.concat(chunks, axis=0)
            
            self.head = torch.from_numpy(df['h'].values)
            self.rel = torch.from_numpy(df['r'].values)
            self.tail = torch.from_numpy(df['t'].values)
            
            self.num_entities = max(df['h'].max(), df['t'].max()) + 1
            self.num_relations = df['r'].max() + 1
            
            print(f"[SUCCESS] Dataset loaded.")
            print(f"[INFO] Triples: {len(df):,}")
            print(f"[INFO] Entities: {self.num_entities:,}")
            print(f"[INFO] Relations: {self.num_relations}")
            
            if self.num_entities > 5_000_000:
                print(f"[WARN] Large entity count ({self.num_entities:,}). May cause OOM.")
            
            del df, chunks
            gc.collect()
            
        except Exception as e:
            print(f"[ERROR] Loading failed: {e}")
            raise e

    def __len__(self):
        return len(self.head)

    def __getitem__(self, idx):
        return {
            'head': self.head[idx],
            'relation': self.rel[idx],
            'tail': self.tail[idx]
        }

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.models import Generator, Discriminator

DATA_PATH = Path("data/processed/kg_triples_ids.txt")
SYNTHETIC_DIR = Path("data/synthetic")
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_PATH = CHECKPOINT_DIR / "gan_latest.pth"
LOG_FILE = Path("data/processed/training_log.csv")

EMBEDDING_DIM = 128   
BATCH_SIZE = 2048   
HIDDEN_DIM = 256
MAX_EPOCHS = 1000
EPOCHS_PER_RUN = 1
LR = 0.00005  
CLIP_VALUE = 0.01  
device = torch.device("cpu") 

def train():
    if not DATA_PATH.exists():
        print(f"[ERROR] Data file not found: {DATA_PATH}")
        return

    dataset = DBLPDataset(DATA_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    print(f"[INFO] Initializing model. Entities: {dataset.num_entities:,}, DIM: {EMBEDDING_DIM}")
    
    gc.collect()
    
    try:
        G = Generator(EMBEDDING_DIM, HIDDEN_DIM, dataset.num_relations).to(device)
        D = Discriminator(dataset.num_entities, dataset.num_relations, EMBEDDING_DIM, HIDDEN_DIM).to(device)
        print(f"[SUCCESS] Model initialized.")
    except RuntimeError as e:
        print(f"[CRITICAL] OOM Error. Reduce entity count or embedding dim. Error: {e}")
        return

    optimizer_G = optim.RMSprop(G.parameters(), lr=LR)
    optimizer_D = optim.RMSprop(D.parameters(), lr=LR)

    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    if CHECKPOINT_PATH.exists():
        print(f"[INFO] Loading checkpoint...")
        try:
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
            G.load_state_dict(checkpoint["G_state"])
            D.load_state_dict(checkpoint["D_state"])
            start_epoch = int(checkpoint.get("epoch", 0))
        except: 
            print("[WARN] Checkpoint error, starting fresh.")
            
    if not LOG_FILE.exists():
        with open(LOG_FILE, "w") as f: f.write("Epoch,D_Loss,G_Loss\n")

    end_epoch = min(start_epoch + EPOCHS_PER_RUN, MAX_EPOCHS)
    
    print(f"--- Starting: Epoch {start_epoch+1} ---")

    for epoch in range(start_epoch, end_epoch):
        total_d, total_g, g_updates = 0, 0, 0
        
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Ep {epoch+1}", ncols=80)
        
        for i, batch in pbar:
            h = batch['head'].to(device)
            r = batch['relation'].to(device)
            t = batch['tail'].to(device)      
            
            optimizer_D.zero_grad()
            d_real = D(h, r, D.get_entity_embedding(t)).mean()
            fake_emb = G(torch.randn(h.size(0), EMBEDDING_DIM).to(device), r).detach()
            d_fake = D(h, r, fake_emb).mean()
            d_loss = -(d_real - d_fake)
            d_loss.backward()
            optimizer_D.step()
            for p in D.parameters(): p.data.clamp_(-CLIP_VALUE, CLIP_VALUE)
            total_d += d_loss.item()

            if i % 5 == 0:
                optimizer_G.zero_grad()
                gen_fake = G(torch.randn(h.size(0), EMBEDDING_DIM).to(device), r)
                g_loss = -D(h, r, gen_fake).mean()
                g_loss.backward()
                optimizer_G.step()
                total_g += g_loss.item()
                g_updates += 1
            
            if i % 2000 == 0 and i > 0:
                torch.save({
                    "G_state": G.state_dict(),
                    "D_state": D.state_dict(),
                    "epoch": epoch
                }, CHECKPOINT_PATH)
            
            if i % 1000 == 0:
                gc.collect()
        
        pbar.close()

        avg_d = total_d / len(dataloader)
        avg_g = total_g / max(1, g_updates)
        print(f"Epoch {epoch+1} | D: {avg_d:.5f} | G: {avg_g:.5f}")
        
        with open(LOG_FILE, "a") as f:
            f.write(f"{epoch+1},{avg_d:.6f},{avg_g:.6f}\n")

    print("[INFO] Saving checkpoint...")
    torch.save({"G_state": G.state_dict(), "D_state": D.state_dict(), "epoch": end_epoch}, CHECKPOINT_PATH)
    print("[SUCCESS] Checkpoint saved.")
    
    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
    with open(SYNTHETIC_DIR / "generated.txt", "w") as f:
        f.write("HEAD\tREL\tTAIL\tSCORE\n")
        f.write("gen\t0\tgen\t0.0\n")
    print("[INFO] Placeholder generation file created.")

if __name__ == "__main__":
    train()