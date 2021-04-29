from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import groupby

from typing import List

EOL = "<EOL>"
EOB = "<EOB>"


class MolBlock(ABC):
    def __init__(self, text: str):
        self.text = text

    @abstractmethod
    def compress(self) -> List[str]:
        pass

    @classmethod
    @abstractmethod
    def from_compressed(cls, compressed_block: List[str]):
        pass

    @classmethod
    def decompress_multiline(cls, compressed_block):
        mol_string = ""
        curr_string = []
        while len(compressed_block) != 0:
            el = compressed_block.pop(0)
            if el != EOL:
                curr_string.append(el)
            else:
                mol_string += " ".join(curr_string) + "\n"
                curr_string = []
        return cls(mol_string[:-1])


@dataclass
class MolCountsLine(MolBlock):
    text: str

    def compress(self) -> List[str]:
        compressed = []
        elements = self.text.split()
        compressed.extend(elements[:2])
        compressed.append(elements[4])
        compressed.append(EOL)
        return compressed

    @classmethod
    def from_compressed(cls, compressed_block: List[str]):
        elements = compressed_block
        return cls(
            " ".join(
                [
                    elements[0],
                    elements[1],
                    "0",
                    "0",
                    elements[2],
                    "0",
                    "0",
                    "0",
                    "0",
                    "0999",
                    "V2000",
                ]
            )
        )


@dataclass
class MolAtomBlock(MolBlock):
    text: str

    def compress(self) -> List[str]:
        compressed = []
        for string in self.text.split("\n"):
            elements = string.split()
            compressed.append(elements[0])
            compressed.append(elements[6])
            compressed.append(EOL)
        return compressed

    @classmethod
    def from_compressed(cls, compressed_block: List[str]):
        mol_string = ""
        curr_string = []
        while len(compressed_block) != 0:
            el = compressed_block.pop(0)
            if el != EOL:
                curr_string.append(el)
            else:
                mol_string += (
                    curr_string[0] + 5 * " 0" + " " + curr_string[1] + 6 * " 0" + "\n"
                )
                curr_string = []
        return cls(mol_string[:-1])


@dataclass
class MolBondBlock(MolBlock):
    text: str

    def compress(self) -> List[str]:
        compressed = []
        for string in self.text.split("\n"):
            elements = string.split()
            compressed.extend(elements[:4])
            compressed.append(EOL)
        return compressed

    @classmethod
    def from_compressed(cls, compressed_block: List[str]):
        return cls.decompress_multiline(compressed_block)


@dataclass
class MolFourthBlock(MolBlock):
    text: str

    def compress(self) -> List[str]:
        compressed = []
        for string in self.text.split("\n"):
            for element in string.split():
                compressed.append(element)
            compressed.append(EOL)
        return compressed

    @classmethod
    def from_compressed(cls, compressed_block: List[str]):
        return cls.decompress_multiline(compressed_block)


@dataclass
class MolFile:
    counts_line: MolCountsLine
    atom_block: MolAtomBlock
    bond_block: MolBondBlock
    fourth_block: MolFourthBlock

    def compress(self) -> List[str]:
        comp_cl = self.counts_line.compress()
        comp_cl.append(EOB)

        comp_ab = self.atom_block.compress()
        comp_ab.append(EOB)

        comp_bb = self.bond_block.compress()
        comp_bb.append(EOB)

        comp_fb = self.fourth_block.compress()
        comp_fb.append(EOB)

        return comp_cl + comp_ab + comp_bb + comp_fb

    @staticmethod
    def _make_string_block(groupped_strings, block_number):
        string_molecule = ""
        if block_number == 6:
            if len(groupped_strings) == 8:
                return ""
            else:
                for i in range(6, len(groupped_strings) - 2):
                    for element in groupped_strings[i]:
                        string_molecule += " ".join(element) + "\n"
        else:
            for element in groupped_strings[block_number]:
                if block_number == 4:
                    element = element[3:]
                string_molecule += " ".join(element) + "\n"
        return string_molecule[:-1]

    @staticmethod
    def _split_mol_blocks(text):
        curr_molecule = []
        for element in text.split("\n"):
            curr_molecule.append(list(element.split()))

        grouped_strings = [
            list(items) for length, items in groupby(curr_molecule, key=len)
        ]

        txt1_ = MolFile._make_string_block(grouped_strings, 3)
        txt2_ = MolFile._make_string_block(grouped_strings, 4)
        txt3_ = MolFile._make_string_block(grouped_strings, 5)
        txt4_ = MolFile._make_string_block(grouped_strings, 6)

        return txt1_, txt2_, txt3_, txt4_

    @classmethod
    def from_text(cls, mol_text: str):
        txt1, txt2, txt3, txt4 = MolFile._split_mol_blocks(mol_text)
        cl = MolCountsLine(txt1)
        ab = MolAtomBlock(txt2)
        bb = MolBondBlock(txt3)
        fb = MolFourthBlock(txt4)
        return cls(cl, ab, bb, fb)

    @classmethod
    def from_compressed(cls, compressed_mol: str):
        block_indx = 1
        cl_block = []
        ab_block = []
        bb_block = []
        fb_block = []
        for element in compressed_mol:
            if element != EOB:
                if block_indx == 1:
                    cl_block.append(element)
                if block_indx == 2:
                    ab_block.append(element)
                if block_indx == 3:
                    bb_block.append(element)
                if block_indx == 4:
                    fb_block.append(element)
            else:
                block_indx += 1
        return cls(
            MolCountsLine.from_compressed(cl_block),
            MolAtomBlock.from_compressed(ab_block),
            MolBondBlock.from_compressed(bb_block),
            MolFourthBlock.from_compressed(fb_block),
        )
