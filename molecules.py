import uuid
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Tuple, Dict

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from cairosvg import svg2png
from rdkit import Chem
from rdkit.Chem.inchi import MolFromInchi
from rdkit.Chem.Draw import rdMolDraw2D
from svgpathtools import svg2paths


@dataclass
class Category:
    id: int
    name: str
    color: str
    metadata: Dict[str, str] = field(default_factory=dict)
    keypoint_colors: List[str] = field(default_factory=list)
    supercategory: str = ''


@dataclass
class ImageCOCO:
    id: int
    file_name: str
    width: int
    height: int


CATEGORIES = {
    'atom': Category(1, 'atom', '#ef703f'),
    'ring': Category(2, 'ring', '#38fb5c'),
    'SINGLE': Category(3, 'SINGLE', '#e17282'),
    'DOUBLE': Category(4, 'DOUBLE', '#e17282'),
    'TRIPLE': Category(5, 'TRIPLE', '#e17282'),
}


def inchi_to_mol(inchi: str):
    return MolFromInchi(inchi)


def mol_to_svg(mol, size=(400, 400), bond_length=27):
    d2d = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
    opts = d2d.drawOptions()
    opts.fixedBondLength = bond_length
    opts.bondLineWidth = 1
    opts.setHighlightColour((0.,0.,0.,0.))
    opts.setSymbolColour((0.,0.,0.,0.))
    # opts.fontFile = '/usr/share/fonts/truetype/tlwg/Sawasdee.ttf'
    d2d.DrawMolecule(mol)
    d2d.FinishDrawing()
    return d2d.GetDrawingText()


def save_svg(svg: str, svg_out: Path):
    svg_out.parent.mkdir(parents=True, exist_ok=True)
    with open(str(svg_out.absolute()), 'w') as f:
        f.write(svg)


def svg_to_png(svg_path: Path, png_path: Path):
    png_path.parent.mkdir(parents=True, exist_ok=True)
    svg2png(url=str(svg_path.absolute()), write_to=str(png_path.absolute()))


def _merge_bboxes(elements: List[Tuple[int, Tuple[float]]]):
    elem_map = {}
    for name, bbox in elements:
        if name in elem_map:
            bbox_ = elem_map[name]
            elem_map[name] = (
                min(bbox_[0], bbox[0]),
                max(bbox_[1], bbox[1]),
                min(bbox_[2], bbox[2]),
                max(bbox_[3], bbox[3]),
            )
        else:
            elem_map[name] = bbox
    return elem_map


def get_svg_features(svg_path: Path):
    elem_bbox = [(attrs.get('class'), path.bbox())
                 for path, attrs in zip(*svg2paths(str(svg_path.absolute()))) if attrs.get('class')]
    atoms = _merge_bboxes([(int(name.replace('atom-', '')), bbox)
                           for name, bbox in elem_bbox if name.startswith('atom')])
    bonds = _merge_bboxes([(int(name.partition(' ')[0].replace('bond-', '')), bbox)
                           for name, bbox in elem_bbox if name.startswith('bond')])
    return atoms, bonds


def _draw_rectangle(color: Tuple[int, int, int], thickness: int, img: np.ndarray, bbox: Tuple[float]):
    cv2.rectangle(img,
                  (int(bbox[0]), int(bbox[2])),
                  (int(bbox[1]), int(bbox[3])),
                  color,
                  thickness)
    sub = img[int(bbox[2]):int(bbox[3]) + 1, int(bbox[0]):int(bbox[1]) + 1]

    black = np.zeros_like(sub)
    black[0:black.shape[0], 0:black.shape[1]] = color

    blend = cv2.addWeighted(sub, 0.75, black, 0.25, 0)

    img[int(bbox[2]):int(bbox[3]) + 1, int(bbox[0]):int(bbox[1]) + 1] = blend


def draw_annotations(atoms: List[Tuple[float]],
                     rings: List[Tuple[float]],
                     bonds_with_class: List[Tuple[Tuple[float], str]],
                     path: Path,
                     img_path: Path):
    img = cv2.imread(str(img_path.absolute()))
    for atom in atoms:
        _draw_rectangle(
            (255, 0, 0),
            1,
            img,
            atom
        )
    for ring in rings:
        _draw_rectangle(
            (0, 255, 0),
            1,
            img,
            ring
        )
    for bond, clazz in bonds_with_class:
        if clazz == 'AROMATIC':
            _draw_rectangle(
                (0, 128, 128),
                1,
                img,
                bond
            )
        else:
            _draw_rectangle(
                (0, 0, 255),
                1,
                img,
                bond
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path.absolute()), img)


def bbox_to_coco(bbox: Tuple[float], image_id: int, category_id: int, color: str):
    x1, x2, y1, y2 = bbox
    bbox = [
        int(x1),
        int(y1),
        int(x2 - x1) + 1,
        int(y2 - y1) + 1,
    ]
    segm_box = [x1, y1, x2, y1, x2, y2, x1, y2]
    return {
        'id': int(str(uuid.uuid4().int)[:6]),
        'image_id': image_id,
        'category_id': category_id,
        'bbox': bbox,
        'segmentation': [segm_box],
        'area': (x2 - x1) * (y2 - y1),
        'score': 1.,
        "iscrowd": False,
        "isbbox": True,
        "color": color,
        "keypoints": [],
        "metadata": {},
    }


def prepare_coco(atoms, rings, bonds_with_class, name, image_id):
    img = ImageCOCO(image_id, f"{name}.png", 400, 400)
    annotations = []
    for atom in atoms:
        annotations.append(
            bbox_to_coco(atom, image_id, CATEGORIES['atom'].id, CATEGORIES['atom'].color)
        )
    for ring in rings:
        annotations.append(
            bbox_to_coco(ring, image_id, CATEGORIES['ring'].id, CATEGORIES['ring'].color)
        )
    for bbox, clazz in bonds_with_class:
        category = CATEGORIES.get(clazz)
        if not category:
            CATEGORIES[clazz] = Category(len(CATEGORIES) + 1, clazz, '#e17282')
            category = CATEGORIES[clazz]
        annotations.append(
            bbox_to_coco(bbox, image_id, category.id, category.color)
        )
    return asdict(img), annotations


def to_binary_img(img):
    img = img.copy()
    if len(img.shape) == 3 and img.shape[2] > 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bg = img[0][0]
    mask = img == bg
    img[mask] = 255
    mask = img != 255
    img[mask] = 0
    return img


def add_noise(img, color, percent):
    out_img = img.copy()
    noise = np.random.rand(*out_img.shape[:2])
    noise_mask = noise > percent
    out_img[noise_mask] = color
    return out_img


def apply_binary_threshold(img, threshold):
    img = img.copy()
    ret,thresh1 = cv2.threshold(img, threshold,255,cv2.THRESH_BINARY)
    return thresh1


def binary_cappification(img):
    img = img.copy()
    binary_img = to_binary_img(img)
    small = cv2.resize(binary_img, (400, 400), cv2.INTER_NEAREST)
    small = add_noise(small, 255, 0.8)
    small = add_noise(small, 0, 0.999)
    small = cv2.dilate(small,(2,2),iterations = 1)
    big = cv2.resize(small, (400, 400), cv2.INTER_NEAREST)
    binary_big = apply_binary_threshold(big, 160)
    return binary_big


def provide_crappificated(png_path: Path, crappy_path: Path):
    img = cv2.imread(str(png_path.absolute()))
    img = binary_cappification(img)
    crappy_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(crappy_path.absolute()), img)


def process_single(inchi: str, out_root: Path, name: str, img_id: int):
    mol = inchi_to_mol(inchi)
    try:
        Chem.Kekulize(mol, clearAromaticFlags=True)
        Chem.SanitizeMol(mol)
        mol = rdMolDraw2D.PrepareMolForDrawing(mol, kekulize=True)
    except Exception as e:
        print(e)
    svg_text = mol_to_svg(mol)
    save_svg(svg_text, out_root / 'svg' / f"{name}.svg")
    svg_to_png(out_root / 'svg' / f"{name}.svg", out_root / 'png' / f"{name}.png")
    provide_crappificated(out_root / 'png' / f"{name}.png", out_root / 'crappy' / f"{name}.png")
    atoms, bonds = get_svg_features(out_root / 'svg' / f"{name}.svg")
    mol_bonds = mol.GetBonds()
    bonds_with_class = []
    for b_id, bbox in bonds.items():
        try:
            bonds_with_class.append((bbox, str(mol_bonds[b_id].GetBondType())))
        except IndexError:
            print(f"Out of bounds {name}")
    ring_info = mol.GetRingInfo()
    bond_rings = ring_info.BondRings()

    rings = []
    for ring in bond_rings:
        coords = list(zip(*[bonds[b_id] for b_id in ring]))
        rings.append((
            min(coords[0]),
            max(coords[1]),
            min(coords[2]),
            max(coords[3]),
        ))

    atoms = [bbox for key, bbox in atoms.items()]

    draw_annotations(atoms, rings, bonds_with_class, out_root / 'ann' / f"{name}.png", out_root / 'png' / f"{name}.png")
    return prepare_coco(atoms, rings, bonds_with_class, name, img_id)


def process_split(path: Path, split: str, name_inchis: List[List[str]]):
    root_split_path = path / split
    root_split_path.mkdir(exist_ok=True, parents=True)
    imgs = []
    annotations = []
    for _idx, name, inchi in name_inchis:
        img_coco, ann_coco = process_single(inchi, root_split_path, name, int(str(uuid.uuid4().int)[:6]))
        imgs.append(img_coco)
        annotations.extend(ann_coco)
    return imgs, annotations


def save_coco(imgs, ann, categories, dataset_root, split):
    coco = {
        'images': imgs,
        'annotations': ann,
        'categories': [asdict(c) for c in categories.values()],
    }
    filepath = dataset_root / split / f"{split}.json"
    with open(str(filepath.absolute()), "w") as f:
        f.write(json.dumps(coco))


def split_dataset(dataset_root: Path, csv_path: Path, limit=3000):
    sample = pd.read_csv(str(csv_path.absolute())).head(limit)
    other, test = train_test_split(sample, test_size=0.1)
    train, val = train_test_split(other, test_size=0.2)
    train_imgs, train_ann = process_split(dataset_root, 'train', train.values.tolist())
    val_imgs, val_ann = process_split(dataset_root, 'val', val.values.tolist())
    test_imgs, test_ann = process_split(dataset_root, 'test', test.values.tolist())
    categories = CATEGORIES

    save_coco(train_imgs, train_ann, categories, dataset_root, 'train')
    save_coco(val_imgs, val_ann, categories, dataset_root, 'val')
    save_coco(test_imgs, test_ann, categories, dataset_root, 'test')


def main():
    split_dataset(
        Path('/home/egor/Desktop/molecule-recognizer/split'),
        Path('/home/egor/Desktop/molecule-recognizer/datasets/sample_train_dataset/train_sample_dataset.csv'),
        1000
    )


def trash():
    inchi = 'InChI=1S/C54H90O6/c1-4-7-10-13-16-19-22-25-26-27-30-32-35-38-41-44-47-53(56)59-50-51(60-54(57)48-45-42-39-36-33-29-24-21-18-15-12-9-6-3)49-58-52(55)46-43-40-37-34-31-28-23-20-17-14-11-8-5-2/h7,10,16,19,25-26,28-29,31,33,37,39-40,42,51H,4-6,8-9,11-15,17-18,20-24,27,30,32,34-36,38,41,43-50H2,1-3H3/b10-7-,19-16-,26-25-,31-28-,33-29-,40-37-,42-39-'
    mol = MolFromInchi(inchi)
    inchi = 'InChI=1S/C15H19ClO2/c1-15(8-4-3-5-9-15)14(17)11-6-7-13(18-2)12(16)10-11/h6-7,10H,3-5,8-9H2,1-2H3'
    inchi = 'InChI=1S/C17H10BrN3O/c18-14-7-3-1-5-11(14)9-12(10-19)16-20-15-8-4-2-6-13(15)17(22)21-16/h1-9,13H/b12-9+'
    inchi = 'InChI=1S/C25H34N2O7/c1-15-4-5-19-16(2)22(31-23-25(19)18(15)8-11-24(3,32-23)33-34-25)30-21(29)7-6-20(28)27-14-17-9-12-26-13-10-17/h9-10,12-13,15-16,18-19,22-23H,4-8,11,14H2,1-3H3,(H,27,28)/t15-,16-,18+,19+,22-,23-,24+,25+/m1/s1'
    mol = MolFromInchi(inchi)
    process_single(inchi, Path('/home/ilia/mol_set'), 'inchi_3', uuid.uuid4().int)


if __name__ == '__main__':
    main()
