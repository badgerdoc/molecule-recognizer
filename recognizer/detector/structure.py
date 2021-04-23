from dataclasses import dataclass

from typing import List, Optional

from recognizer.common.boxes import BoundaryBox
from recognizer.common.constants import ATOM_CLS, DOUBLE_CLS, RING_CLS, \
    SINGLE_CLS, \
    TRIPLE_CLS


@dataclass
class StructureElement:
    bbox: BoundaryBox


@dataclass
class Atom(StructureElement):
    name: Optional[str] = None


@dataclass
class Bond(StructureElement):
    number: int


@dataclass
class Ring(StructureElement):
    pass


@dataclass
class DetectedStructure:
    atoms: List[Atom]
    bonds: List[Bond]
    rings: List[Ring]

    @classmethod
    def from_bboxes_list(cls, bboxes):
        atoms = []
        bonds = []
        rings = []
        for box in bboxes:
            bbox = BoundaryBox(*box['bbox'])
            label = box['label']
            if label == ATOM_CLS:
                atoms.append(Atom(bbox))
            elif label == RING_CLS:
                rings.append(Ring(bbox))
            elif label is SINGLE_CLS:
                bonds.append(Bond(bbox, 1))
            elif label is DOUBLE_CLS:
                bonds.append(Bond(bbox, 2))
            elif label is TRIPLE_CLS:
                bonds.append(Bond(bbox, 3))
            else:
                raise ValueError(f'Unknown class "{label}"')
        return cls(atoms, bonds, rings)
