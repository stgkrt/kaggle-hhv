import numpy as np

# https://www.kaggle.com/code/paulorzp/run-length-encode-and-decode/script


def rle_encode(img):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    """
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    rle = " ".join(str(r) for r in runs)
    if rle == "":
        rle = "1 0"
    return rle


def rle_decode(mask_rle: str, shape: tuple) -> np.array:
    """
    Decode rle string
    https://www.kaggle.com/code/paulorzp/run-length-encode-and-decode/script
    https://www.kaggle.com/stainsby/fast-tested-rle

    Args:
      mask_rle: run length (rle) as string
      shape: (height, width) of the mask

    Returns:
      array[uint8], 1 - mask, 0 - background
    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)
