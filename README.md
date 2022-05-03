# Environment setup

Clone the repo:
`git clone https://github.com/cmcin019/AV_Image_Analysis.git`

1. Install conda for Linux
	```
	wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
	bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda
	$HOME/miniconda/bin/conda init bash
	```
2. Create conda environment
	```
	conda env create -f conda_env.yml
	conda activate AV_ENV
	
	cd AV_Image_Analysis
	```

3. Create checkpoints folder 
	```
	mkdir checkpoints
	```

4. Download checkpoints from add them to the checkpoints folder
	PSPNET
	https://drive.google.com/file/d/1ydBpkFAZ0CX7BD3iALGTr03UUUd_sfXn/view?usp=sharing 
	
	DEPTH
	https://drive.google.com/file/d/13XS8X5p_mS_-SBuEsw1Sw7yGVUvYg0tY/view?usp=sharing 


5. Download big lama folder and add it to the repository
	Big-LAMA
	https://drive.google.com/drive/folders/17l_wIZhZ-YWi_MUtCiz4164TnNr96Xxx?usp=sharing

6. Install mmcv and depth library
	```
	pip install mmcv-full==1.3.13 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html
	pip install -e .
	```

7. Save path
	```
	export TORCH_HOME=$(pwd) && export PYTHONPATH=.
	```

# Run 

Compute SEGMENTATION and INPAINTING videos 
```
python inpainting_test.py model.path=$(pwd)
```
Compute DEPTH video 
```
python tools/test.py configs/depthformer/depthformer_swinl_22k_w7_kitti.py \
    checkpoints/depthformer_swinl_22k_kitti.pth \
    --show-dir road_depth
```
    
Compute road detection example
```
python road_detect_test.py
```

# Acknowledgments
* Segmentation form [MMSegmentation](https://github.com/open-mmlab/mmsegmentation).
* Inpainting from [LAMA](https://github.com/saic-mdal/lama)
* Depth Estimation from [Monocular-Depth-Estimation](https://github.com/zhyever/Monocular-Depth-Estimation-Toolbox)

