'''
生成masks
'''

from typing import Tuple
import numpy as np

COLORS = ((244,  67,  54),
          (233,  30,  99),
          (156,  39, 176),
          (103,  58, 183),
          ( 63,  81, 181),
          ( 33, 150, 243),
          (  3, 169, 244),
          (  0, 188, 212),
          (  0, 150, 136),
          ( 76, 175,  80),
          (139, 195,  74),
          (205, 220,  57),
          (255, 235,  59),
          (255, 193,   7),
          (255, 152,   0),
          (255,  87,  34),
          (121,  85,  72),
          (158, 158, 158),
          ( 96, 125, 139))

def assignColor(idx):
    num_colors = len(COLORS)
    idx %= num_colors
    color = COLORS[idx]
    hex_color = f'#{hex(color[0])}{hex(color[1])}{hex(color[2])}'
    hex_color = hex_color.replace('0x', '')
    return hex_color

class MaskMakerBase(object):
    name_classes = []
    def __init__(self, name) -> None:
        self.name = name
        if name not in MaskMakerBase.name_classes:
            MaskMakerBase.name_classes.append(name)

    def __call__(self)->Tuple:
        raise NotImplemented

    @classmethod
    def classes_info(self):
        classes_info_list = []
        for id, name in enumerate(MaskMakerBase.name_classes, 0):   # 缺陷id从0开始
            class_info_dict = {
                "attribute_groups": [],
                "color": assignColor(id),
                "id": id + 1,
                "name": name,
                "opened": True
            }
            classes_info_list.append(class_info_dict)
        return classes_info_list

# 多边形
class PolygonMaskMaker(MaskMakerBase):
    def __init__(self, 
                name,
                aradius, 
                bradius, 
                num_vertex = 8, 
                random_ratote = True,
                ratio = 0.75
                ) -> None:
        super().__init__(name)
        self.aradius = aradius
        self.bradius = bradius
        self.num_vertex = num_vertex
        self. random_ratote = random_ratote  # 是否随机角度旋转
        self.ratio = ratio

    def __call__(self, img_size)->Tuple:  # (classId, points)
        classId = MaskMakerBase.name_classes.index(self.name)
        maxx, maxy = img_size[0], img_size[1]
        x_, y_ = np.random.randint(0, maxx), np.random.randint(0, maxy)   # 随机生成缺陷位置
        a, b = self.aradius, self.bradius
        '''
        根据椭圆方程生成多边形
        x = acos(θ)
        y = bcos(θ)
        ''' 
        rotate_angle = np.random.randint(0, 360)
        α = np.deg2rad(rotate_angle)
        points = []
        θs = np.linspace(0, 2*np.pi, self.num_vertex)
        for θ in θs:
            a_ = np.random.uniform(a* self.ratio, a / self.ratio)
            b_ = np.random.uniform(b* self.ratio, b / self.ratio)
            _x = a_*np.cos(θ)
            _y = b_*np.sin(θ)
            x,y=0,0
            if self.random_ratote:
                x = _x*np.cos(α) - _y*np.sin(α) + x_
                y = _x*np.sin(α) + _y*np.cos(α) + y_
            if x >= 0 and x <= maxx and y >=0 and y <= maxy:
                points.append(x)
                points.append(y)

        return classId, points

# 点状
class DotMaskMaker(PolygonMaskMaker):
    def __init__(self) -> None:
        super().__init__('Dot', 5, 5, ratio = 0.9)

# 线状
class LineMaskMaker(PolygonMaskMaker):
    def __init__(self) -> None:
        super().__init__('Line', 18, 1.5, 20, 0.8, ratio=0.8)
    
# 块状
class BlockMaskMaker(PolygonMaskMaker):
    def __init__(self) -> None:
        super().__init__('Block', 12, 12, 10, ratio=0.75)