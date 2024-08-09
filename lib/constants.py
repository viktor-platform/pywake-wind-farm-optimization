import base64
from pathlib import Path

import numpy as np
from viktor import File

ROOT = Path(__file__).parent.parent
IMAGE_DPI = 800
ENCODING = "utf-8"


def get_divisors(n, minimum=5):
    """
    Returns a list of divisors of `n` that are greater than or equal to `minimum`.
    """
    bin_nums = np.arange(minimum, n + 1)
    remainders = np.remainder(n, bin_nums)
    mask = remainders == 0
    return bin_nums[mask].tolist()


def serialize(file: File) -> str:
    """
    Encodes a File object to a base64-encoded string.
    """
    return base64.b64encode(file.getvalue_binary()).decode(encoding=ENCODING)


def deserialize(s: str) -> File:
    """
    Decodes a base64-encoded string to a File object.
    """
    return File.from_data(base64.b64decode(s.encode(encoding=ENCODING)))

