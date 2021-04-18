from pathlib import Path

from cairosvg import svg2png
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from svgpathtools import svg2paths
from typing import Optional, List, Tuple


def mol_to_svg(mol, size=(400, 400), bond_length=27, save_path: Optional[Path]=None, bw_mode=True):
    d2d = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
    opts = d2d.drawOptions()
    opts.fixedBondLength = bond_length
    opts.bondLineWidth = 1
    opts.setHighlightColour((0., 0., 0., 0.))
    opts.setSymbolColour((0., 0., 0., 0.))
    if bw_mode:
        opts.useBWAtomPalette()
    # opts.fontFile = '/usr/share/fonts/truetype/tlwg/Sawasdee.ttf'
    d2d.DrawMolecule(mol)
    d2d.FinishDrawing()
    text = d2d.GetDrawingText()
    if save_path is not None:
        with open(save_path, 'w') as f:
            f.write(text)
    return text


def mol_to_png(mol, write_to: Path):
    try:
        Chem.Kekulize(mol, clearAromaticFlags=True)
        Chem.SanitizeMol(mol)
        mol = rdMolDraw2D.PrepareMolForDrawing(mol, kekulize=True)
    except Exception as e:
        print(e)
    svg_text = mol_to_svg(mol)
    svg2png(bytestring=svg_text, write_to=str(write_to.absolute()))


def get_svg_elements(svg_path: Path):
    return [(attrs.get('class'), path.bbox())
             for path, attrs in zip(*svg2paths(str(svg_path.absolute()))) if attrs.get('class')]


def get_atoms_from_svg_elements(elem_bbox):
    return _merge_bboxes([(int(name.replace('atom-', '')), bbox)
                           for name, bbox in elem_bbox if name.startswith('atom')])


def find_atom_bboxes(svg_path: Path):
    elems = get_svg_elements(svg_path)
    return get_atoms_from_svg_elements(elems)


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
