import re
import math

import cv2
import io
import numpy as np
from pathlib import Path

from xml.dom import minidom

from cairosvg import svg2png
from rdkit.Chem import MolFromInchi
from typing import Tuple, Optional, List, Dict

from recognizer.common.molecule_utils import mol_to_svg, _merge_bboxes

SVG_COORD_REGEXP = r'((\d+\.?\d*)\s(\d+\.?\d*))'


class MoleculeImageGenerator:

    def __init__(self, add_padding=True):
        self.add_padding = add_padding

    def inchi_to_image(
        self,
        inchi: str,
        img_size: Tuple[int, int],
        save_path: Optional[Path] = None,
        bond_length: int = 27,
        bw_mode=True
    ):
        mol = MolFromInchi(inchi)
        svg_text = mol_to_svg(
            mol, size=img_size, bond_length=bond_length, save_path=save_path,
            bw_mode=bw_mode
        )
        mol_svg = MoleculeSVG(svg_text)
        img_stream = io.BytesIO()
        img_stream.write(svg2png(bytestring=svg_text))
        img_stream.seek(0)
        img = cv2.imdecode(np.fromstring(img_stream.read(), np.uint8), 1)
        atom_bboxes = mol_svg.get_atom_bboxes()
        if self.add_padding:
            img = self._create_padding_around_atoms(img, atom_bboxes)
        return img

    @staticmethod
    def _create_padding_around_atoms(img, atom_bboxes: Dict[int, Tuple[float]], thk=2):
        """Ensure that there is distance between bonds and letters."""
        for box in atom_bboxes.values():
            x1, x2, y1, y2 = [int(v) for v in box]
            x1 = x1 - thk
            y1 = y1 - thk
            x2 = x2 + thk
            y2 = y2 + thk
            img = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), thickness=thk)
        return img


class MoleculeSVG:
    def __init__(self, svg_text: str):
        self.svg = minidom.parseString(svg_text)

    @property
    def paths(self):
        return self.svg.getElementsByTagName('path')

    def get_svg_elements(self):
        return [(_get_class(p), _get_bbox(p)) for p in self.paths]

    def get_atom_bboxes(self, merge_atoms=True) -> List[Tuple[int, int, int, int]]:
        elems = self.get_svg_elements()
        atom_boxes = [
            (int(name.replace('atom-', '')), bbox) for name, bbox in elems
            if name.startswith('atom')
        ]
        if merge_atoms:
            atom_boxes = _merge_bboxes(atom_boxes)
        return atom_boxes


def _get_class(p):
    return p.getAttribute('class')


def _get_bbox(p):
    box_str: str = p.getAttribute('d')
    p = re.compile(SVG_COORD_REGEXP)
    matches = re.findall(p, box_str)
    coords = [(float(m[1]), float(m[2])) for m in matches]
    x1, x2 = math.inf, -1
    y1, y2 = math.inf, -1
    for x, y in coords:
        y1 = y if y < y1 else y1
        y2 = y if y > y2 else y2
        x1 = x if x < x1 else x1
        x2 = x if x > x2 else x2
    return x1, x2, y1, y2
