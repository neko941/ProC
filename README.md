# DeLUR
<!-- ## Update System
```
sudo apt-get update
sudo apt-get upgrade 
sudo apt update
sudo apt upgrade 
sudo apt install gcc
```
## Install NVIDIA drivers
## Remove previous NVIDIA installation
```shell
sudo apt autoremove nvidia* --purge
```

### Check Ubuntu devices
```shell
ubuntu-drivers devices
```
You will install the NVIDIA driver whose version is tagged with __recommended__


### Install Ubuntu drivers
```shell
sudo ubuntu-drivers autoinstall
```

### Install NVIDIA drivers
My __recommended__ version is 525, adapt to yours

```shell
sudo apt install nvidia-driver-525
```
```shell
sudo ubuntu-drivers autoinstall
sudo apt-get install build-essential python3-dev
sudo apt-get install -y cuda-drivers
sudo apt install ubuntu-drivers-common
```
### Reboot & Check
```shell
reboot
```
after restart verify that the following command works
```shell
nvidia-smi
```

## Install CUDA drivers
https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_network
```shell
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda
sudo apt-get -y install cuda-toolkit-12-4
```
### Check CUDA install
```shell
nvcc --version
```

## Install cuDNN
https://developer.nvidia.com/cudnn-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_network
```shell
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cudnn
``` -->

```
python -c 'import torch; print(torch.version.cuda)'
nvcc --version
```
should be the same version
Check Pytorch 
```
python -c 'import torch; print(f"{torch.cuda.is_available() = }"); print(f"{torch.cuda.device_count() = }"); print(f"{torch.cuda.current_device() = }"); [print(f"Device {i}: {torch.cuda.get_device_name(i)}") for i in range(torch.cuda.device_count())]'
```
## Virtual Environment
### Windows
```bash
pip install virtualenv
python -m virtualenv venv --python=3.10.10
.\venv\Scripts\activate
```
### Linux
```
source venv/bin/activate
```
## Requirements
```bash
# pip3 install torch==1.12 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```
<!-- ```
pip install --upgrade pip setuptools wheel
https://github.com/Dao-AILab/causal-conv1d
cd causal-conv1d
export FORCE_CUDA="1"
MAX_JOBS=2 CAUSAL_CONV1D_FORCE_BUILD="TRUE" python setup.py bdist_wheel --dist-dir=dist
python setup.py install
``` -->