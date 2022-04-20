'''
将subannotation数据集转换为coco数据集
'''
from encodings import utf_8
from fileinput import filename
from operator import sub
from pathlib import Path
import json
import PIL.Image as Image
from segmentdataset import SegmentDataset
import os, shutil

def create_info():
    return {
        'description': 'COCO Dataset from subannotation', 
        'url': 'http://cocodataset.org', 
        'version': '1.0', 
        'year': 2022, 
        'contributor': 'cjf', 
        'date_created': '2022/4/7'
    }

def create_licenses():
    return []

def create_images(subdataset:SegmentDataset, save_imgs_dir):
    images = []
    # 图像标注信息
    for id, img_annos in enumerate(subdataset.imgs_annos, 0):
        path = list(img_annos)[0]
        if not Path(path).exists():
            continue
        filename = Path(path).resolve().name
        shutil.copyfile(path, save_imgs_dir.joinpath(filename))  # 复制图像到images目录下面

        # 图像信息
        img = Image.open(path)
        info = {
            'licenses':-1,
            'file_name':filename,
            'coco_url':'',
            'height':img.height,
            'width':img.width,
            'date_captured':"",
            'flickr_url':'',
            'id': id
        }
        images.append(info)
    return images

def create_annotations(subdataset:SegmentDataset):
    annotations = []
    idx = 0
    # 返回轮廓的最小正切矩形
    def box(points):
        xs, ys = points[0::2],points[1::2]
        min_x, max_x, min_y, max_y = min(xs)-1, max(xs)-1, min(ys)-1, max(ys)-1  # 外扩一个像素
        return [min_x, min_y, max_x-min_x, max_y-min_y]

    for img_id, img_annos in enumerate(subdataset.imgs_annos, 0):
        annos = list(img_annos.items())[0][1]
        for anno in annos:
            classId, points = list(anno.items())[0]
            obj = {
                'segmentation':None,
                'area':-1,
                'iscrowd':0,
                'image_id':img_id,
                'bbox':box(points),   # [x,y,w,h]
                'category_id':classId + 1,   # 背景类id为0
                'id':idx,
            }
            obj['segmentation'] = [points]
            annotations.append(obj)
            idx += 1
    return annotations

def create_categories(subdataset:SegmentDataset):
    class_infos = []
    classIds = []
    for img_annos in subdataset.imgs_annos:
        annos = list(img_annos.items())[0][1]
        for anno in annos:
            anno = list(anno.items())[0]
            classId = anno[0] + 1
            if not set(classIds).__contains__(classId):
                class_info = {
                    'subpercategory':'obj',
                    'id':classId,
                    'name':str(classId)
                }
                if class_info not in class_infos:
                    class_infos.append(class_info)
    return class_infos

def subdataset2coco(subdataset:SegmentDataset, save_dir):
    subdataset.init_annotaions()

    if not Path(save_dir).exists():
        os.mkdir(save_dir)

    # 创建文件夹
    imgs_dir = Path(save_dir).joinpath('images')
    annos_dir = Path(save_dir).joinpath('annotations')
    if not Path(imgs_dir).exists():
        os.mkdir(imgs_dir)
    if not Path(annos_dir).exists():
        os.mkdir(annos_dir)

    # anno文件大体格式
    content = {
        "info": None, # dict
        "licenses": None, # list ，内部是dict,可以为空
        "images": None, # list ，内部是dict
        "annotations": None, # list ，内部是dict
        "categories": None, # list ，内部是dict
    }

    # 创建info
    content['info'] = create_info()
    # 创建licenses
    content['licenses'] = create_licenses()
    # 创建images信息
    content['images'] = create_images(subdataset, imgs_dir)
    # 创建annotations信息
    content['annotations'] = create_annotations(subdataset)
    # 创建categories信息
    content['categories'] = create_categories(subdataset)

    # 保存文件
    annos_file = Path(save_dir).joinpath('annotations/segments.json')
    try:
        with open(annos_file, 'w', encoding='utf_8') as json_file:
            json.dump(content, json_file)
        print('proccessed successed!')
    except:
        print('proccessed failed!')

if __name__ == '__main__':
    from subannodataset import SubAnnoDataset
    w,h = 608,608
    num = 100
    subdataset = SubAnnoDataset(f'./data/testdata-{w}-{h}-{num}/')
    subdataset2coco(subdataset, f'./data/testdata-{w}-{h}-{num}/coco-{w}-{h}')