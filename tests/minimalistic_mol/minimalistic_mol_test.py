import pytest
import json
from recognizer.common.minimalistic_mol import *


@pytest.fixture
def sample_mol_file() -> str:
    with open("./resources/test_min_mol.json", "r") as f:
        test = json.load(f)
    return test


def test_compress(sample_mol_file):
    gt_test_1_cl = ["25", "28", "0", EOL]
    gt_test_1_ab = ["C", "0", EOL, "C", "0", EOL, "C", "0", EOL, "C", "0", EOL,
                    "C", "0", EOL, "C", "0", EOL, "C", "0", EOL, "C", "0", EOL,
                    "C", "0", EOL, "C", "0", EOL, "C", "0", EOL, "C", "0", EOL,
                    "C", "0", EOL, "C", "0", EOL, "C", "0", EOL, "C", "0", EOL,
                    "C", "0", EOL, "C", "0", EOL, "C", "0", EOL, "C", "0", EOL,
                    "C", "0", EOL, "O", "0", EOL, "O", "0", EOL, "O", "0", EOL,
                    "O", "0", EOL]
    gt_test_1_bb = [ "1", "12", "1", "0", EOL, "20",  "2", "1", "1", EOL,
                    "21",  "3", "1", "1", EOL,  "4",  "5", "1", "0", EOL,
                    "15",  "4", "1", "1", EOL,  "5", "18", "1", "0", EOL,
                     "6",  "8", "1", "0", EOL,  "6", "14", "1", "0", EOL,
                     "7",  "9", "1", "0", EOL, "16",  "7", "1", "6", EOL,
                     "8", "20", "1", "0", EOL,  "9", "21", "1", "0", EOL,
                    "13", "10", "1", "1", EOL, "10", "14", "1", "0", EOL,
                    "11", "13", "1", "0", EOL, "11", "17", "1", "0", EOL,
                    "12", "22", "2", "0", EOL, "12", "25", "1", "0", EOL,
                    "13", "20", "1", "0", EOL, "14", "25", "1", "1", EOL,
                    "15", "19", "1", "0", EOL, "15", "21", "1", "0", EOL,
                    "16", "19", "1", "0", EOL, "16", "20", "1", "0", EOL,
                    "19", "17", "1", "1", EOL, "17", "23", "2", "0", EOL,
                    "18", "21", "1", "0", EOL, "18", "24", "2", "0", EOL]
    gt_test_1_fb = [EOL]

    gt_test_2_cl = ["14", "13", "0", EOL]
    gt_test_2_ab = [ "C", "0", EOL,  "C", "0", EOL, "C", "0", EOL, "C", "0", EOL,
                     "C", "0", EOL,  "C", "0", EOL, "N", "0", EOL, "N", "0", EOL,
                     "O", "0", EOL,  "O", "0", EOL, "O", "3", EOL, "S", "6", EOL,
                    "Si", "3", EOL, "Si", "2", EOL]
    gt_test_2_bb = [ "1",  "3", "1", "0", EOL,  "2", "13", "1", "0", EOL,
                     "3", "12", "1", "0", EOL,  "4",  "6", "1", "0", EOL,
                     "4",  "7", "3", "0", EOL,  "5",  "6", "1", "0", EOL,
                     "5",  "8", "3", "0", EOL,  "6", "12", "1", "0", EOL,
                     "6", "14", "1", "0", EOL,  "9", "12", "2", "0", EOL,
                    "10", "12", "2", "0", EOL, "11", "13", "2", "3", EOL,
                    "11", "14", "1", "0", EOL]
    gt_test_2_fb = ["M", "CHG", "2", "11", "1", "13", "-1", EOL,
                    "M", "RAD", "1", "14", "3", EOL]

    for i in ["8", "10"]:
        compr = MolFile.from_text(sample_mol_file[i]).compress()
        block_indx = 1
        cl_block = []
        ab_block = []
        bb_block = []
        fb_block = []
        for element in compr:
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

        if i == "8":
            assert cl_block == gt_test_1_cl
            assert ab_block == gt_test_1_ab
            assert bb_block == gt_test_1_bb
            assert fb_block == gt_test_1_fb

        elif i == "10":
            assert cl_block == gt_test_2_cl
            assert ab_block == gt_test_2_ab
            assert bb_block == gt_test_2_bb
            assert fb_block == gt_test_2_fb


def test_from_compress(sample_mol_file):
    for i in sample_mol_file.keys():
        gt_test = MolFile.from_text(sample_mol_file[i])
        mb = MolFile.from_compressed(MolFile.from_text(sample_mol_file[i]).compress())

        assert gt_test.counts_line.text == mb.counts_line.text
        assert gt_test.atom_block.text == mb.atom_block.text
        assert gt_test.bond_block.text == mb.bond_block.text
        assert gt_test.fourth_block.text == mb.fourth_block.text
