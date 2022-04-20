'''
将subannotation转换为KolektorSDD数据集
'''

from sklearn.utils import shuffle
from segmentdataset import SegmentDataset
from pathlib import Path
import os
import torchvision.transforms as Trans
import numpy as np
from PIL import Image
import cv2
img1 = cv2.imread('./data/1.png')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.imread('./data/2.png')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
imgs = [img1, img2]

def subdataset2kk(subdataset: SegmentDataset, save_dir:str):
    img_size = subdataset.image_size
    print('proccessing...')

    for idx, (img, box) in enumerate(subdataset, 1):

        save_imgs_path = Path(save_dir).joinpath(str(idx))
        if not save_imgs_path.exists():
            os.makedirs(save_imgs_path)

        print(f'proccessing...{idx}/{len(subdataset)}')
        semantic_mask = np.zeros([img_size[1], img_size[0]])
        img = Trans.ToPILImage()(img)
        rect, masks,_ = box
        for i, mask in enumerate(masks):
            semantic_mask += mask*255
        
        for i in range(8):
            if i % 2  == np.random.randint(0, 2):
                img.save(save_imgs_path.joinpath(f'{i}.jpg'))
                save_masks_path = save_imgs_path.joinpath(f'{i}_label.bmp')
                semantic_mask_img = Image.fromarray(semantic_mask).convert('L')
                semantic_mask_img.save(save_masks_path)
            else:
                im = imgs[np.random.randint(0,2)]
                im = cv2.resize(im, img_size)
                cv2.imwrite(str(save_imgs_path.joinpath(f'{i}.jpg')), im)
                save_masks_path = save_imgs_path.joinpath(f'{i}_label.bmp')
                semantic_mask_ = np.zeros([img_size[1], img_size[0]])
                semantic_mask_img = Image.fromarray(semantic_mask_ * 255).convert('L')
                semantic_mask_img.save(save_masks_path)

    print('finished.')

if __name__ == '__main__':
    from subannodataset import SubAnnoDataset
    w, h = 500, 1263
    subdataset = SubAnnoDataset('D:/work/DeepCode/yolact-master/data/unet-testdata-500-1263-50', [w, h])
    subdataset2kk(subdataset, 'D:/work/DeepCode/yolact-master/data/unet-testdata-500-1263-50/kk-50')
