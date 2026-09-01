import torch
import shutil
from pathlib import Path

folder_path = "models"
directory = Path(folder_path)
pt_files = list(directory.rglob("*.pt"))

if not pt_files:
    print("⚠️ No .pt files found in this folder.")
    exit(1)

print(f"Found {len(pt_files)} .pt file(s). Starting conversion...")

for pt_file in pt_files:
    if "_float" in pt_file.name:
        print(f"⚠️ Skipping already converted file: {pt_file.name}")
        continue

    new_path = pt_file.with_name(pt_file.name.replace(".pt", "_float32.pt"))
    
    ckpt = torch.load(pt_file.absolute(), map_location="cpu", weights_only=False)
    ckpt['model'] = ckpt['model'].float()
    torch.save(ckpt, new_path.absolute())
    
    print(f"✅ Converted: {pt_file} -> {new_path}")
