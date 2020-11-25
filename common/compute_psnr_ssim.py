'''
Created on Jul 11, 2019

@author: gsq
'''
import numpy as np
import os
from skimage import measure, color, io

upscale = 4
mode = 'YCbCr'
dataset = 'McMaster'
HRpath = '../CLSR/TestDatasets/{}'.format(dataset)
SRpath = 'results/{}_ronet_x{}'.format(dataset, upscale)
Logpath = os.path.join(SRpath, 'psnr_ssim.txt')
if os.path.exists(Logpath):
    os.remove(Logpath)

def imgCrop(img_path, upscale):
    img = io.imread(img_path)
    if mode == 'YCbCr':
        img = color.rgb2ycbcr(img)
    height = img.shape[0] - img.shape[0] % upscale
    width = img.shape[1] - img.shape[1] % upscale
    if mode == 'YCbCr':
        return img[0:height, 0:width, 0]
    else:
        return img[0:height, 0:width, :]

def shave(img, upscale):
    if mode == 'YCbCr':
        img = img[upscale:img.shape[0]-upscale, upscale:img.shape[1]-upscale]
    else:
        img = img[upscale:img.shape[0]-upscale, upscale:img.shape[1]-upscale, :]
    return img

HRnames = sorted(os.listdir(HRpath))
SRnames = sorted(os.listdir(SRpath))
psnr_list = []
ssim_list = []

for i in range(0, len(HRnames)):
    HRimage = imgCrop(os.path.join(HRpath, HRnames[i]), upscale)
    SRimage = io.imread(os.path.join(SRpath, SRnames[i]))
    if mode == 'YCbCr':
        SRimage = color.rgb2ycbcr(SRimage)
        HR = shave(HRimage, upscale)
        SR = shave(SRimage[:, :, 0], upscale)
        psnr = measure.compare_psnr(HR, SR, data_range=255)
        ssim = measure.compare_ssim(HR, SR, data_range=255, multichannel=True)
    else:
        HR = shave(HRimage, upscale)
        SR = shave(SRimage, upscale)
        psnr = measure.compare_psnr(HR, SR, data_range=255)
        ssim = measure.compare_ssim(HR, SR, data_range=255, multichannel=True)
    print('PSNR:%0.04f   SSIM:%0.04f' % (psnr, ssim))
    psnr_list.append(psnr)
    ssim_list.append(ssim)
    with open(Logpath, 'a') as f:
        f.write('%s    %0.04f    %0.04f \n' % (SRnames[i], psnr, ssim))
average_psnr = np.mean(np.asarray(psnr_list))
average_ssim = np.mean(np.asarray(ssim_list))
print('Mean PSNR: %0.02f  Mean SSIM: %0.04f' % (average_psnr, average_ssim))
with open(Logpath, 'a') as f:
    f.write('%s    %0.04f    %0.04f \n' % ('Average', average_psnr, average_ssim))

