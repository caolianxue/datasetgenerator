'''
采用随机方式生成缺陷，保存为subannotation数据格式
'''
import cv2 as cv
import numpy as np
from pathlib import Path
import os, json
from maskmaker import *
import pycocotools.mask as maskUtils
import PIL.Image as Image

class FlawMaker(object):
    def __init__(self, 
                num_imgs:int,
                genrator_funs,
                save_dir:str,
                bkcolor_func = None, 
                forecolor_func = None, 
                vague_edge = False,
                image_size = [550,550],
                min_num_obj = 1,
                max_num_obj = 5,
                ) -> None:
        self.num_imgs = num_imgs               # 生成图像的数量           
        self.genrator_funs = genrator_funs     # 产生缺陷函数列表
        self.save_dir_path = Path(save_dir)    # 保存目录
        self.bk_func = bkcolor_func            # 产生背景函数
        self.fore_func = forecolor_func        # 产生前景函数
        self.vague_edge = vague_edge           # 模糊边缘
        self.image_size = image_size           # 图像大小
        self.min_num_obj = min_num_obj         # 最少缺陷数量
        self.max_num_obj = max_num_obj         # 最大缺陷数量

    def create_config_files(self):
        # 创建保存目录
        if not self.save_dir_path.exists():
            os.makedirs(self.save_dir_path)

        # 创建images目录
        if not self.save_dir_path.joinpath('images').exists():
            os.makedirs(self.save_dir_path.joinpath('images'))

        # 创建thumb目录
        if not self.save_dir_path.joinpath('images/thumb').exists():
            os.makedirs(self.save_dir_path.joinpath('images/thumb'))

        # 创建config.json文件
        config_path = self.save_dir_path.joinpath('config.json')
        config_content = {"pathSeparator":"\\","os":"win32"}
        with open(config_path, 'w', encoding='utf-8') as handle:
            json.dump(config_content, handle)

        # 创建classes.json文件
        classes_path = self.save_dir_path.joinpath('classes.json')
        classes_content = MaskMakerBase.classes_info()
        with open(classes_path, 'w', encoding='utf-8') as handle:
            json.dump(classes_content, handle)

    # 返回轮廓的mask
    def points2mask(self, points, imgSize):
        _points = [points]
        w, h = imgSize[0], imgSize[1]
        rles = maskUtils.frPyObjects(_points, h, w)
        rle = maskUtils.merge(rles)
        m = maskUtils.decode(rle)
        return m

    def __call__(self):
        # 创建配置文件
        self.create_config_files()

        # 创建图像和annotations.json文件
        img_dir_path = self.save_dir_path.joinpath('images')
        annotations_content = {
            "___sa_version___": "1.0.0"
        }

        # images.sa文件
        saes_content = []

        print('creating...')
        for i in range(self.num_imgs):
            # 生成背景图像
            img = None
            if self.bk_func is None:
                img = np.ones(self.image_size, np.uint8) * 100
            else:
                img = self.bk_func(self.image_size)

            img_path = img_dir_path.joinpath(f'{i}.jpg')
            instances = []
            annotations_img = {
                'instances':instances,
                'tags': [],
                'metadata': {
                'version': '1.0.0',
                'name': img_path.name,
                'status': 'In progress'
                }
            }
            annotations_content[img_path.name] = annotations_img

            # sa
            thumb_path = img_dir_path.joinpath('thumb').joinpath(f'thmb_{img_path.name}')
            sa = {
                "srcPath": str(img_path.absolute()),
                "name": img_path.name,
                "imagePath": str(img_path.absolute()),
                "thumbPath": str(thumb_path.absolute()),
                "valid": True
            }
            saes_content.append(sa)

            num_obj = np.random.randint(self.min_num_obj, self.max_num_obj+1)
            func_idxs = []
            for _ in range(num_obj):
                func_idx = np.random.randint(0, len(self.genrator_funs))
                func_idxs.append(func_idx)
            func_idxs = np.sort(func_idxs)
            
            img_avg_gray = np.mean(img)
            ks = [9, 3, 3]
            for func_idx in func_idxs:
                # 生成annotations和mask
                generator_func = self.genrator_funs[func_idx]
                classId, points = generator_func(self.image_size)
                if len(points) < 2*3:  # 保证至少三个顶点
                    continue
                # 点信息
                instance = {
                    "type": "polygon",
                    "classId": classId + 1,
                    "probability": 100,
                    "points": list(points),
                    "groupId": 0,
                    "pointLabels": {},
                    "locked": False,
                    "visible": True,
                    "attributes": []
                }
                instances.append(instance)
                mask = self.points2mask(points, self.image_size)
                indexs = (mask == 1)
                # gray_v = 100 if self.fore_func is None else self.fore_func()
                coffi = [-1,1][np.random.randint(0,2)]
                gray_v = img_avg_gray + coffi*np.random.randint(60,80)
                if self.vague_edge:
                    indexs = (mask > 0)
                    avg_img = np.ones_like(img) * img_avg_gray
                    mask[indexs] = gray_v
                    avg_img[indexs] = mask[indexs]
                    k = ks[func_idx]
                    avg_img = cv.GaussianBlur(avg_img, ksize=(k, k), sigmaX=0, sigmaY=0)
                    mask[indexs] = avg_img[indexs]
                else:
                    mask[indexs] = gray_v
                img[indexs] = mask[indexs]

            # 写入图像
            PIL_img = Image.fromarray(img)
            PIL_img = PIL_img.convert('L')
            PIL_img.save(img_path)

            # 缩略图
            PIL_img = PIL_img.resize((96,96),Image.ANTIALIAS)
            PIL_img.save(thumb_path)

            print(f'creating... {i+1}/{self.num_imgs}')

        # annotations.json文件
        annotations_path = self.save_dir_path.joinpath('annotations.json')
        with open(annotations_path, 'w') as handle:
            json.dump(annotations_content, handle)

        # 创建imges.sa文件
        sa_path = img_dir_path.joinpath('images.sa')
        with open(sa_path, 'w') as handle:
            json.dump(saes_content, handle)

        print('finished.')

# 随机灰度
class rand_gray(object):
    def __init__(self, min, max) -> None:
        self.min = min
        self.max = max
    def __call__(self):
        return np.random.randint(self.min, self.max)

import cv2
img1 = cv2.imread('1.png')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.imread('2.png')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
imgs = [img1, img2]

# 生成灰度图像
class rand_gray_img(object):
    def __init__(self, min, max) -> None:
        self.min = min
        self.max = max
    def __call__(self, img_size):
        gray_value = np.random.randint(self.min, self.max)
        return np.ones(img_size) * gray_value

    # 生成灰度图像
class rand_gray_img2(object):
    def __init__(self, img_size) -> None:
        self.images = []
        for im in imgs:
            _im = cv2.resize(im, img_size)
            self.images.append(_im)
    def __call__(self, img_size):
        idx = np.random.randint(0, len(self.images))
        return self.images[idx].copy()

if __name__ == '__main__':
    w,h = 500,1263
    num = 50
    flawMaker = FlawMaker(num, 
                        [BlockMaskMaker(), LineMaskMaker(), DotMaskMaker()],   #  排序，依次从下到上，避免小目标被遮挡
                        f'./data/testdata-{w}-{h}-{num}',
                        bkcolor_func = rand_gray_img2([w,h]),
                        forecolor_func= rand_gray(30,225),
                        image_size=[w,h],
                        vague_edge=True,
                        min_num_obj=1,
                        max_num_obj=3
                        )
    flawMaker()