'''
将subannotation数据集转换为unet数据集
'''
from sklearn.utils import shuffle
from segmentdataset import SegmentDataset
from pathlib import Path
import os
import torchvision.transforms as Trans
import numpy as np
from PIL import Image

def subdataset2unet(subdataset: SegmentDataset, save_dir:str):
    save_imgs_dir = Path(save_dir).joinpath('images')
    save_masks_dir = Path(save_dir).joinpath('masks')
    if not save_imgs_dir.exists():
        os.makedirs(save_imgs_dir)
    if not save_masks_dir.exists():
        os.makedirs(save_masks_dir)

    num_classes = subdataset.get_num_classes() - 1  # 去除背景类别
    img_size = subdataset.image_size

    print('proccessing...')
    for idx, (img, box) in enumerate(subdataset, 1):
        print(f'proccessing...{idx}/{len(subdataset)}')
        semantic_masks = np.zeros([num_classes, img_size[0], img_size[1]])
        img = Trans.ToPILImage()(img)
        rect, masks,_ = box
        for i, mask in enumerate(masks):
            classid = int(rect[i][-1])
            semantic_masks[classid,:,:] += mask
        img.save(save_imgs_dir.joinpath(f'{idx}.png'))
        for classid, semantic_mask in enumerate(semantic_masks):
            save_masks_classes_dir = save_masks_dir.joinpath(str(classid))
            if not save_masks_classes_dir.exists():
                os.makedirs(save_masks_classes_dir)
            save_masks_path = save_masks_classes_dir.joinpath(f'{idx}.png')
            semantic_mask_img = Image.fromarray(semantic_mask * 255).convert('L')
            semantic_mask_img.save(save_masks_path)
    print('finished.')

if __name__ == '__main__':
    from subannodataset import SubAnnoDataset
    s = 480
    subdataset = SubAnnoDataset(f'./data/unet-testdata-{s}-300/', [s,s])
    subdataset2unet(subdataset, f'./data/unet-testdata-{s}-300/subannos-{s}')