# How to train on RADEON RX 6750 XT

## 0. Use python 3.12.x

`sudo dnf install python3.12 python3.12-devel`

## 1. Pip VENV

`python3.12 -m venv .venv`

## 2. Install pytorch for ROCm

`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.1`

Check with:

`python -c "import torch; print('PyTorch:', torch.__version__); print('HIP:', torch.version.hip); print('GPU available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"`

If not working retry with:

`export HSA_OVERRIDE_GFX_VERSION=10.3.0`

## 3. Install ultralytics

`pip install ultralytics`

## 4. Run training with parameters

`HSA_OVERRIDE_GFX_VERSION=10.3.0 \
AMD_SERIALIZE_KERNEL=3 \
python train.py`
