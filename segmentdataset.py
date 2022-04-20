'''
分割数据集
'''
import numpy as np
import torch
import torch.utils.data.dataset as dataset
import PIL.Image as Image
import torchvision.transforms as Trans
import pycocotools.mask as maskUtils

'''
数据集格式如下：
[
    'img_path1':[
        classId:x0,y0,x1,y1,...
        classId:x0,y0,x1,y1,...
        ....
        ]

    'img_path2':[
        classId:x0,y0,x1,y1,...
        classId:x0,y0,x1,y1,...
        ....
        ]
]
'''
class SegmentDataset(dataset.Dataset):
    def __init__(self, img_size = [500,500]) -> None:
        super().__init__()
        self.imgs_annos = None
        self.image_size = img_size

    def init_annotaions(self):
        raise NotImplementedError

    def __len__(self):
        if self.imgs_annos is None:
            self.init_annotaions()
        return len(self.imgs_annos)

    def __getitem__(self, index):
        if self.imgs_annos is None:
            self.init_annotaions()
        img_annos = list(self.imgs_annos[index].items())[0]
        path, annos = img_annos[0], img_annos[1]
        img = Image.open(path)
        if len(img.split()) == 4:
            img = img.convert("RGB")
        _img = img.resize(self.image_size)

        trans = Trans.Compose([
            Trans.ToTensor()
        ])
        img_tensor = trans(_img)
        if img_tensor.shape[0] == 1:
            img_tensor = torch.vstack([img_tensor, img_tensor, img_tensor])

        boxes, masks = [], []
        for i, anno in enumerate(annos):
            anno = list(anno.items())[0]
            classId, points = anno[0], anno[1]
            counter = trans_points(img.size, self.image_size, points)
            # show_annos(_img, self.image_size, counter)
            box = boxofcounter(counter, classId, self.image_size)
            # show_box(_img, box)
            mask = counter2mask(counter, self.image_size)
            # show_mask(mask)
            boxes.append(box)
            masks.append(mask)

        boxes = np.asarray(boxes)
        masks = np.asarray(masks)

        return img_tensor, (boxes, masks, 0)

    def get_num_classes(self):
        if self.imgs_annos is None:
            self.init_annotaions()
        classIds = []
        for img_annos in self.imgs_annos:
            annos = list(img_annos.items())[0][1]
            for anno in annos:
                anno = list(anno.items())[0]
                classId = anno[0]
                if not set(classIds).__contains__(classId):
                    classIds.append(classId)
        return len(classIds) + 1  # 包括背景类

# 轮廓点矫正
def trans_points(srcSize, desSize, points):
    points = list(points)
    sx = desSize[0]/srcSize[0]
    sy = desSize[1]/srcSize[1]
    xs = [x*sx for x in points[0::2]]
    ys = [y*sy for y in points[1::2]]
    return np.array([xs, ys])

# 返回轮廓的最小正切矩形
def boxofcounter(counter, classId = None, imgSize = None):
    xs, ys = counter
    w ,h = (1, 1) if imgSize is None else (imgSize[0], imgSize[1])
    min_x, max_x, min_y, max_y = (min(xs)-1)/w, (max(xs)-1)/w, (min(ys)-1)/h, (max(ys)-1)/h  # 外扩一个像素
    return np.array([min_x, min_y, max_x, max_y, classId])   

# 返回轮廓的mask
def counter2mask(counter, imgSize):
    w, h = imgSize[0], imgSize[1]
    xs, ys = counter[0], counter[1]
    points = np.zeros(2*len(xs))
    points[0::2] = xs
    points[1::2] = ys
    points = points.reshape([1, -1])
    _points = []
    _points.append(list(points[0]))
    rles = maskUtils.frPyObjects(_points, h, w)
    rle = maskUtils.merge(rles)
    m = maskUtils.decode(rle)
    return m

def show_annos(img, imgSize, counter):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    plt.cla()
    img = img.resize(imgSize)
    plt.imshow(img)
    xs, ys = counter[0::2], counter[1::2]
    xs = np.hstack(xs)
    ys = np.hstack(ys)
    _points = []
    for x, y in zip(xs, ys):
        _points.append([x, y])
    polygon = Polygon(_points, True, color='red', alpha = 0.5)
    plt.gca().add_patch(polygon)
    plt.show()

def show_mask(mask):
    import matplotlib.pyplot as plt
    plt.cla()
    plt.imshow(mask)
    plt.show()

def show_box(img, box):
    import matplotlib.pyplot as plt
    plt.cla()
    plt.imshow(img)
    w, h = img.width, img.height
    x1,y1,x2,y2 = box[0]*w,box[1]*h,box[2]*w,box[3]*h
    rect = plt.Rectangle((x1,y1),x2-x1,y2-y1,color='red',alpha=0.5)
    plt.gca().add_patch(rect)
    plt.show()