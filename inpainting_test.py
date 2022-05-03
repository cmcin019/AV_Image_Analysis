from mmseg.apis import inference_segmentor, init_segmentor
import mmcv
import cv2 
import os
import shutil
import warnings

import logging
import sys
import traceback

from saicinpainting.evaluation.utils import move_to_device

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import cv2
import hydra
import numpy as np
import torch
import tqdm
import yaml
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate

from saicinpainting.training.data.datasets import make_default_val_dataset
from saicinpainting.training.trainers import load_checkpoint as saicinpainting_load_checkpoint
from saicinpainting.utils import register_debug_signal_handlers

from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmcv.utils import DictAction

#from depth.apis import single_gpu_test
#from depth.datasets import build_dataloader, build_dataset
#from depth.models import build_depther

LOGGER = logging.getLogger(__name__)


config_file = 'configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py'
checkpoint_file = 'checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'

model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

dir_list = os.listdir('data/')

dir_list.sort()
os.system('cls' if os.name == 'nt' else 'clear')

img = cv2.imread('data/' + dir_list[0])

height, width = img.shape[:-1]

out = cv2.VideoWriter('videos/original_001.avi',cv2.VideoWriter_fourcc(*'DIVX'), 8, (width, height))
f = open("data/kitti/test_split.txt", "w")

print('Segmentation process')
for x in tqdm.tqdm(dir_list):

	if ('png' in x):
		result = inference_segmentor(model, 'data/' + x)
		img = cv2.imread('data/' + x, 1)
		cv2.imwrite('road_seg/' + x, img) #use path here
		model.show_result(img, result, out_file='road_seg/'+x[:-4]+'_mask.png', opacity=1)
		
		f.write("2011_09_26/2011_09_26_drive_0002_sync/image_02/data/"+x[:-4]+'_mask.png'+" 2011_09_26_drive_0002_sync/proj_depth/groundtruth/image_02/0000000069.png\n")
		out.write(img)
		
		pass
	pass
	
f.close()
out.release()

@hydra.main(config_path='configs/prediction', config_name='default.yaml')
def main(predict_config: OmegaConf):
    try:

        register_debug_signal_handlers()  # kill -10 <pid> will result in traceback dumped into log

        device = torch.device(predict_config.device)

        predict_config.model.path += '/big-lama'
        train_config_path = os.path.join(predict_config.model.path, 'config.yaml')
        with open(train_config_path, 'r') as f:
            train_config = OmegaConf.create(yaml.safe_load(f))
        
        train_config.training_model.predict_only = True
        train_config.visualizer.kind = 'noop'

        out_ext = predict_config.get('out_ext', '.png')

        checkpoint_path = os.path.join(predict_config.model.path, 
                                       'models', 
                                       predict_config.model.checkpoint)
        model = saicinpainting_load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')
        model.freeze()
        model.to(device)

        predict_config.indir = predict_config.model.path[:-9] + predict_config.indir
        if not predict_config.indir.endswith('/'):
            predict_config.indir += '/'

        dataset = make_default_val_dataset(predict_config.indir, **predict_config.dataset)
        os.system('cls' if os.name == 'nt' else 'clear')
        print('Inpainting process')

        with torch.no_grad():
            for img_i in tqdm.trange(len(dataset)):
                mask_fname = dataset.mask_filenames[img_i]
                cur_out_fname = os.path.join(
                    predict_config.model.path[:-9] + predict_config.outdir, 
                    os.path.splitext(mask_fname[len(predict_config.indir):])[0] + out_ext
                )
                os.makedirs(os.path.dirname(cur_out_fname), exist_ok=True)

                batch = move_to_device(default_collate([dataset[img_i]]), device)
                batch['mask'] = (batch['mask'] > 0) * 1
                batch = model(batch)
                cur_res = batch[predict_config.out_key][0].permute(1, 2, 0).detach().cpu().numpy()

                cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
                cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
                cv2.imwrite(cur_out_fname, cur_res)
                if img_i == 0:
                    height, width = cur_res.shape[:-1]
                    out_s = cv2.VideoWriter(predict_config.model.path[:-9] + '/videos/segmentation_001.avi',cv2.VideoWriter_fourcc(*'DIVX'), 8, (width, height))

                out_s.write(cur_res)
                
        out_s.release()
    except KeyboardInterrupt:
        LOGGER.warning('Interrupted by user')
    except Exception as ex:
        LOGGER.critical(f'Prediction failed due to {ex}:\n{traceback.format_exc()}')
        sys.exit(1)
    
    #os.system('cls' if os.name == 'nt' else 'clear')
    #print('Depth process')
    
    #cfg = mmcv.Config.fromfile('configs/depthformer/depthformer_swinl_22k_w7_kitti.py')
    #cfg.model.pretrained = None
    #cfg.data.test.test_mode = True
    
    #dataset = build_dataset(cfg.data.test)
    #data_loader = build_dataloader(
    #    dataset,
    #    samples_per_gpu=1,
    #    workers_per_gpu=cfg.data.workers_per_gpu,
    #    dist=distributed,
    #    shuffle=False)
    
    ## build the model and load checkpoint
    #cfg.model.train_cfg = None

    #model = build_depther(
    #    cfg.model,
    #    test_cfg=cfg.get('test_cfg'))
    
    #fp16_cfg = cfg.get('fp16', None)
    #if fp16_cfg is not None:
    #    wrap_fp16_model(model)

    #checkpoint = load_checkpoint(model, 'checkpoints/depthformer_swinl_22k_kitti.pth', map_location='cpu')
    #torch.cuda.empty_cache()
        
    #model = MMDataParallel(model, device_ids=[0])
    #results = single_gpu_test(
    #    model,
    #    data_loader,
    #    False,
    #    'depthformer_results_II',
    #    pre_eval=False,
    #    format_only=False,
    #    format_args={})
        
    #rank, _ = get_dist_info()

        
        
if __name__ == '__main__':
    main()








