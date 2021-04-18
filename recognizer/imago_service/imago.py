import subprocess
from pathlib import Path

from typing import Optional


class ImagoService:
    def __init__(self, indigo_binary_path: Path):
        self.indigo_binary_path = indigo_binary_path

    def image_to_mol(self, img_path: Path, mol_path: Path) -> Optional[Path]:
        try:
            subprocess.run(
                [str(self.indigo_binary_path), '-o', str(mol_path), str(img_path)]
            )
        except Exception as e:
            print(e)