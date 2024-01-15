import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from score import _neighbour_code_to_normals
from score.rle_convert import rle_decode

_NEIGHBOUR_CODE_TO_NORMALS = _neighbour_code_to_normals._NEIGHBOUR_CODE_TO_NORMALS

torch_ver_major = int(torch.__version__.split(".")[0])
dtype_index = torch.int32 if torch_ver_major >= 2 else torch.long
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
di = "/kaggle/input/blood-vessel-segmentation"


def compute_area(y: list, unfold: nn.Unfold, area: torch.Tensor) -> torch.Tensor:
    """
    Args:
      y (list[Tensor]): A pair of consecutive slices of mask
      unfold: nn.Unfold(kernel_size=(2, 2), padding=1)
      area (Tensor): surface area for 256 patterns (256, )

    Returns:
      Surface area of surface in 2x2x2 cube
    """
    # Two layers of segmentation masks
    yy = torch.stack(y, dim=0).to(torch.float16).unsqueeze(0)
    # (batch_size=1, nch=2, H, W)
    # bit (0/1) but unfold requires float

    # unfold slides through the volume like a convolution
    # 2x2 kernel returns 8 values (2 channels * 2x2)
    cubes_float = unfold(yy).squeeze(0)  # (8, n_cubes)

    # Each of the 8 values are either 0 or 1
    # Convert those 8 bits to one uint8
    cubes_byte = torch.zeros(cubes_float.size(1), dtype=dtype_index, device=device)
    # indices are required to be int32 or long for area[cube_byte] below, not uint8
    # Can be int32 for torch 2.0.0, int32 raise IndexError in torch 1.13.1.

    for k in range(8):
        cubes_byte += cubes_float[k, :].to(dtype_index) << k

    # Use area lookup table: pattern index -> area [float]
    cubes_area = area[cubes_byte]

    return cubes_area


def create_table_neighbour_code_to_surface_area(spacing_mm):
    """Returns an array mapping neighbourhood code to the surface elements area.

    Note that the normals encode the initial surface area. This function computes
    the area corresponding to the given `spacing_mm`.

    from https://www.kaggle.com/code/metric/surface-dice-metric/

    Args:
      spacing_mm: 3-element list-like structure. Voxel spacing in x0, x1 and x2
        direction.
    """
    # compute the area for all 256 possible surface elements
    # (given a 2x2x2 neighbourhood) according to the spacing_mm
    neighbour_code_to_surface_area = np.zeros([256])
    for code in range(256):
        normals = np.array(_NEIGHBOUR_CODE_TO_NORMALS[code])
        sum_area = 0
        for normal_idx in range(normals.shape[0]):
            # normal vector
            n = np.zeros([3])
            n[0] = normals[normal_idx, 0] * spacing_mm[1] * spacing_mm[2]
            n[1] = normals[normal_idx, 1] * spacing_mm[0] * spacing_mm[2]
            n[2] = normals[normal_idx, 2] * spacing_mm[0] * spacing_mm[1]
            area = np.linalg.norm(n)
            sum_area += area
        neighbour_code_to_surface_area[code] = sum_area

    return neighbour_code_to_surface_area.astype(np.float32)


def compute_surface_dice_score(submit: pd.DataFrame, label: pd.DataFrame) -> float:
    """
    Compute surface Dice score for one 3D volume

    submit (pd.DataFrame): submission file with id and rle
    label (pd.DataFrame): ground truth id, rle, and also image height, width
    """
    # submit and label must contain exact same id in same order
    assert (submit["id"] == label["id"]).all()
    assert len(label) > 0

    # All height, width must be the same
    len(label["height"].unique()) == 1
    len(label["width"].unique()) == 1

    # Surface area lookup table: Tensor[float32] (256, )
    area = create_table_neighbour_code_to_surface_area((1, 1, 1))
    area = torch.from_numpy(area).to(device)  # torch.float32

    # Slide through the volume like a convolution
    unfold = torch.nn.Unfold(kernel_size=(2, 2), padding=1)

    r = label.iloc[0]
    h, w = r["height"], r["width"]
    n_slices = len(label)

    # Padding before first slice
    y0 = y0_pred = torch.zeros((h, w), dtype=torch.uint8, device=device)

    num = 0  # numerator of surface Dice
    denom = 0  # denominator
    for i in range(n_slices + 1):
        # Load one slice
        if i < n_slices:
            r = label.iloc[i]
            y1 = rle_decode(r["rle"], (h, w))
            y1 = torch.from_numpy(y1).to(device)

            r = submit.iloc[i]
            y1_pred = rle_decode(r["rle"], (h, w))
            y1_pred = torch.from_numpy(y1_pred).to(device)
        else:
            # Padding after the last slice
            y1 = y1_pred = torch.zeros((h, w), dtype=torch.uint8, device=device)

        # Compute the surface area between two slices (n_cubes,)
        area_pred = compute_area([y0_pred, y1_pred], unfold, area)
        area_true = compute_area([y0, y1], unfold, area)

        # True positive cube indices
        idx = torch.logical_and(area_pred > 0, area_true > 0)

        # Surface dice numerator and denominator
        num += area_pred[idx].sum() + area_true[idx].sum()  # type: ignore
        denom += area_pred.sum() + area_true.sum()  # type: ignore

        # Next slice
        y0 = y1
        y0_pred = y1_pred

    dice = num / denom.clamp(min=1e-8)  # type: ignore
    return dice.item()


def add_size_columns(df: pd.DataFrame):
    """
    df (DataFrame): including id column, e.g., kidney_1_dense_0000
    """
    widths = []
    heights = []
    subdirs = []
    nums = []
    for i, r in df.iterrows():
        file_id = r["id"]
        subdir = file_id[:-5]  # kidney_1_dense
        file_num = file_id[-4:]  # 0000

        filename = "%s/train/%s/images/%s.tif" % (di, subdir, file_num)
        img = Image.open(filename)
        w, h = img.size
        widths.append(w)
        heights.append(h)
        subdirs.append(subdir)
        nums.append(file_num)

    df["width"] = widths
    df["height"] = heights
    df["image_id"] = subdirs
    df["slice_id"] = nums
