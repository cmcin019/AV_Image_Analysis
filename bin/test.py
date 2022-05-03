from mmseg.apis import inference_segmentor, init_segmentor
import mmcv
import cv2 
import os

config_file = '../configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py'
checkpoint_file = '../checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'

model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

dir_list = os.listdir('../data/')

dir_list.sort()

for x in dir_list:
	if ('png' in x):
		result = inference_segmentor(model, '../data/' + x)
		img = cv2.imread('../data/' + x, 1)
		cv2.imwrite('../output/' + x, img) #use path here
		model.show_result(img, result, out_file='../output/'+x[:-4]+'_mask.png', opacity=1)
		
		
		pass
	pass







