# DeLUR
## Update System
```
sudo apt-get update
sudo apt-get install build-essential python3-dev
sudo apt autoremove
sudo apt-get install -y cuda-drivers
```
## CUDA
```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-4
```
## Virtual Environment
```bash
pip install virtualenv
python -m virtualenv venv --python=3.10.10
.\venv\Scripts\activate
```
```
source venv/bin/activate
```
## Requirements
```bash
pip install --upgrade pip setuptools wheel
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```