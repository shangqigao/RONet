#!/bin/sh

#-----------------------RO decomposition------------------------------#
#train UROD-G (unsupervised RODec for grayscale images)
#nohup python -u RODec_train.py --train_mode unsupervised --out_channel 1 --input_data_dir ../data/Train --augment --log_dir ../logs/UROD-G --GPU_ids 0 >out &

#test UROD-G
#python RODec_test.py --dataset DIV2K_mild --input_data_dir ../data/Test/benchmark --out_channel 1 --DecMethod RODec --RODec_checkpoint ../models/UROD-G/model --sigma 30 --GPU_ids 0 

#train UROD-C (unsupervised RODec for RGB images)
#nohup python -u RODec_train.py --train_mode unsupervised --out_channel 3 --input_data_dir ../data/Train --augment --log_dir ../logs/UROD-C --GPU_ids 0 >out &

#test UROD-C
#python RODec_test.py --dataset DIV2K_mild --input_data_dir ../data/Test/benchmark --out_channel 3 --DecMethod RODec --RODec_checkpoint ../models/UROD-C/model --sigma 30 --GPU_ids 0


#-----------------------RO reconstruction-----------------------------#
#=====================Grayscale image denoising=======================#
#train RONet-G with sigma=50
#nohup python -u RONet_train.py --input_data_dir ../data/Train --augment --task DEN --net_type net_den --deep_scale 48 --depth_RODec 1 --depth_RecROs 3 --depth_RecRes 6 --depth_RecFus 3 --out_channel 1 --RODec_checkpoint ../models/UROD-G/model --sigma 50 --log_dir ../logs/RONet-G --GPU_ids 0 >out &

#test RONet-G with sigma=50
#python RONet_test.py --dataset RNI6 --input_data_dir ../data/Test/benchmark --task DEN --net_type net_den --deep_scale 48 --depth_RODec 1 --depth_RecROs 3 --depth_RecRes 6 --depth_RecFus 3 --out_channel 1 --RONet_checkpoint ../logs/RONet-G_sigma50/model --save_dir ../results --sigma 50 --GPU_ids 2

#=======================RGB image denoising===========================#
#trian RONet-C with sigma range in [0, 75]
#nohup python -u RONet_train.py --input_data_dir ../data/Train --augment --task DEN --net_type net_den --deep_scale 16 --depth_RODec 1 --depth_RecROs 3 --depth_RecRes 6 --depth_RecFus 3 --out_channel 3 --sigma 75 --range --RODec_checkpoint ../models/UROD-C/model --log_dir ../models/RONet-C --GPU_ids 0 >out &

#test RONet-C with sigma=50
#python RONet_test.py --dataset CBSD68 --input_data_dir ../data/Test/benchmark --task DEN --net_type net_den --deep_scale 16 --depth_RODec 1 --depth_RecROs 3 --depth_RecRes 6 --depth_RecFus 3 --out_channel 3 --RONet_checkpoint ../models/RONet-C/model --save_dir ../results --sigma 50 --GPU_ids 0

#===================Bicubic image super-resolution====================#
#train RONet-NF for SR (x4)
#nohup python -u RONet_train.py --input_data_dir ../data/Train --augment --task BiSR --upscale 4 --net_type net_sr --depth_RODec 3 --depth_RecROs 3 --depth_RecRes 6 --depth_RecFus 3 --out_channel 3 --vgg_checkpoint ../models/vgg_19.ckpt --RODec_checkpoint ../models/UROD-C/model --log_dir ../logs/RONet-NF --GPU_ids 0 >out &

#test RONet-NF for SR (x4)
#python RONet_test.py --dataset Set5 --input_data_dir ../data/Test/benchmark --task BiSR --upscale 4 --net_type net_sr --depth_RODec 3 --depth_RecROs 3 --depth_RecRes 6 --depth_RecFus 3 --out_channel 3 --RONet_checkpoint ../models/RONet-NF/model --save_dir ../results --GPU_ids 0

#===================Realistic image super-resolution====================#
#train RONet-R for SR (x4)
#nohup python -u RONet_train.py --input_data_dir ../data/Train --augment --task ReSR --upscale 4 --net_type net_sr --depth_RODec 3 --depth_RecROs 3 --depth_RecRes 6 --depth_RecFus 3 --out_channel 3 --vgg_checkpoint ../models/vgg_19.ckpt --RODec_checkpoint ../models/UROD-C/model --log_dir ../logs/RONet-R --GPU_ids 0 >out &

#test RONet-R for SR (x4)
#python RONet_test.py --dataset DIV2K_mild --input_data_dir ../data/Test/benchmark --task ReSR --upscale 4 --net_type net_sr --depth_RODec 3 --depth_RecROs 3 --depth_RecRes 6 --depth_RecFus 3 --out_channel 3 --RONet_checkpoint ../logs/RONet-R-E2E/model --save_dir ../results --ensemble --GPU_ids 0




