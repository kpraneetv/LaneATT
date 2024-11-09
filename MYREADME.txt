I have reproduced the LaneATT resnet34 on TUSimple dataset experiment on WSL2 Linux 20.4

Follow the commands on the Linux console:

1) sudo apt update && sudo apt upgrade -y
2) sudo apt install -y nvidia-cuda-toolkit
   This installs CUDA 10.1
   verify using "nvcc --version"
3) sudo apt install -y python3 python3-venv python3-pip
   python3 -m venv laneatt-env
   source laneatt-env/bin/activate 
4) pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 torchaudio==0.6.0 -f https://download.pytorch.org/whl/torch_stable.html
5) git clone https://github.com/lucastabelini/LaneATT.git
   cd LaneATT 
6) pip install -r requirements.txt
7) Download the TUSimple dataset. Follow DATASETS.md
8) cd lib\nms 
   python3 setup.py build develop
   this installs nms files
9) Training the model
   python main.py train --exp_name laneatt_r34_tusimple --cfg cfgs/laneatt_tusimple_resnet34.yml
   Once trained experiments/models will have pytorch trained models at every epoch.
10) Follow the README.md for testing commands.