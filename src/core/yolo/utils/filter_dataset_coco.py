from pathlib import Path
from typing import List, Dict
import shutil
import json

# Configuration dictionary
CONFIG: Dict[str, object] = {
    'filter_names': [
        'person', 'bicycle', 'car', 'motorcycle'
    ],  # classes to keep
    'all_names': [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
        'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
        'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ],
    'splits': ['train2017', 'val2017', 'test2017'],  # dataset partitions
    'src_images_root': 'coco/images',
    'src_labels_root': 'coco/labels',
    'src_annotations_root': 'coco/annotations',  # for JSON
    'dest_root': 'coco_filter',
    'tqdm': {
        'enabled': True,
        'desc': 'Filtering {split}',
        'unit': 'file'
    }
}

class CocoFilter:
    def __init__(
        self,
        filter_names: List[str],
        all_names: List[str],
        src_images: Path,
        src_labels: Path,
        src_ann_root: Path,
        dest_images: Path,
        dest_labels: Path,
        dest_ann: Path,
        tqdm_cfg: Dict[str, object]
    ):
        self.filter_names = filter_names
        self.all_names = all_names
        self.filter_indices = [all_names.index(n) for n in filter_names]
        self.filter_cat_ids = [i+1 for i in self.filter_indices]
        self.src_images = src_images
        self.src_labels = src_labels
        self.src_ann_root = src_ann_root
        self.dest_images = dest_images
        self.dest_labels = dest_labels
        self.dest_ann = dest_ann
        self.tqdm_cfg = tqdm_cfg

    def _prepare_dirs(self):
        for d in (self.dest_images, self.dest_labels, self.dest_ann):
            d.mkdir(parents=True, exist_ok=True)

    def filter_split(self, split: str):
        self._prepare_dirs()
        from tqdm import tqdm
        # YOLO filtering and copy
        if self.src_labels.exists():
            txt_list = sorted(self.src_labels.glob('*.txt'))
            iterator = tqdm(txt_list, desc=self.tqdm_cfg['desc'].format(split=split), unit=self.tqdm_cfg['unit']) if self.tqdm_cfg['enabled'] else txt_list
            for txt_path in iterator:
                lines = [l.strip() for l in txt_path.read_text().splitlines() if l.strip()]
                filtered = [ln for ln in lines if int(ln.split()[0]) in self.filter_indices]
                if filtered:
                    out_txt = self.dest_labels / txt_path.name
                    out_txt.write_text("\n".join(filtered)+"\n")
                    img_src = self.src_images / txt_path.with_suffix('.jpg').name
                    if img_src.exists():
                        shutil.copy2(img_src, self.dest_images / img_src.name)
        else:
            jpg_list = sorted(self.src_images.glob('*.jpg'))
            iterator = tqdm(jpg_list, desc=self.tqdm_cfg['desc'].format(split=split), unit=self.tqdm_cfg['unit']) if self.tqdm_cfg['enabled'] else jpg_list
            for img_path in iterator:
                shutil.copy2(img_path, self.dest_images / img_path.name)

        # COCO JSON for val2017
        if split == 'val2017':
            src_json = self.src_ann_root / f'instances_{split}.json'
            if src_json.exists():
                data = json.loads(src_json.read_text())
                anns = [a for a in data['annotations'] if a['category_id'] in self.filter_cat_ids]
                img_ids = {a['image_id'] for a in anns}
                imgs = [im for im in data['images'] if im['id'] in img_ids]
                cats = [c for c in data['categories'] if c['id'] in self.filter_cat_ids]
                out = {k: data[k] for k in ('info','licenses')}
                out.update({'images': imgs, 'annotations': anns, 'categories': cats})
                (self.dest_ann / f'instances_{split}.json').write_text(json.dumps(out, indent=2))


def main():
    cfg = CONFIG
    src_img_root = Path(cfg['src_images_root'])
    src_lbl_root = Path(cfg['src_labels_root'])
    src_ann_root = Path(cfg['src_annotations_root'])
    dest = Path(cfg['dest_root'])
    tqdm_cfg = cfg['tqdm']

    for split in cfg['splits']:
        cf = CocoFilter(
            filter_names=cfg['filter_names'],
            all_names=cfg['all_names'],
            src_images=src_img_root/split,
            src_labels=src_lbl_root/split,
            src_ann_root=src_ann_root,
            dest_images=dest/'images'/split,
            dest_labels=dest/'labels'/split,
            dest_ann=dest/'annotations',
            tqdm_cfg=tqdm_cfg
        )
        cf.filter_split(split)

    # Copy LICENSE and README.txt
    for fname in ('LICENSE','README.txt'):
        if Path(fname).exists():
            shutil.copy2(Path(fname), dest/fname)

    # Generate manifests
    for split in cfg['splits']:
        manifest = 'test-dev2017.txt' if split=='test2017' else f'{split}.txt'
        with open(dest/manifest,'w') as mf:
            for img in sorted((dest/'images'/split).glob('*.jpg')):
                mf.write(f"./images/{split}/{img.name}\n")

    print(f"Finished. Check '{dest}' structure.")

if __name__=='__main__':
    main()