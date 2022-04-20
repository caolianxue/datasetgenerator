'''
subannotation数据集
'''
import sys
sys.path.append('data/')
from segmentdataset import SegmentDataset
from pathlib import Path
import json

class SubAnnoDataset(SegmentDataset):
    def __init__(self, data_dir, img_size = [550,550]) -> None:
        super().__init__(img_size)
        self.data_dir = Path(data_dir).resolve()

    def init_annotaions(self):
        img_dir = self.data_dir.joinpath('images')
        anno_path = self.data_dir.joinpath('annotations.json')
        if not img_dir.exists() or not anno_path.exists():
            return

        self.imgs_annos = []
        with open(anno_path, 'r', encoding='utf_8') as f:  
            content = json.load(f)
            for key, value in content.items():
                img_path = img_dir.joinpath(key)
                if Path(img_path).suffix in ['.bmp', '.jpg', '.png', '.tiff', '.jpge'] and Path(img_path).exists():
                    img_annos = []
                    instances = value['instances']
                    for i, instance in enumerate(instances):
                        classId, xys = instance['classId']-1, instance['points']
                        if len(xys):
                            anno = {}
                            anno[classId] = xys
                            img_annos.append(anno)
                    if len(img_annos):
                        self.imgs_annos.append({str(img_path):img_annos})

if __name__ == '__main__':
    SubAnnoDataset('./data/counter/')[0]
