% Install conda for Linux
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda
$HOME/miniconda/bin/conda init bash

% Create conda environment
conda env create -f conda_env.yml
conda activate AV_ENV

% conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge

% Install mmcv and depth library
pip install mmcv-full==1.3.13 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html
pip install -e .

% Save path
export TORCH_HOME=$(pwd) && export PYTHONPATH=.

% Compute SEGMENTATION and INPAINTING videos 
python inpainting_test.py model.path=$(pwd)

% Compute DEPTH video 
python tools/test.py configs/depthformer/depthformer_swinl_22k_w7_kitti.py \
    checkpoints/depthformer_swinl_22k_kitti.pth \
    --show-dir road_depth
    
% Compute road detection example
python road_detect_test.py
