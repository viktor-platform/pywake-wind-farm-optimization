from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
OPTIMIZED_POSITIONS_PATH = ROOT / "lib" / "optimized_positions_plot.png"
OPTIMIZED_AEP_PATH = ROOT / "lib" / "optimized_positions_aep"
OUT = ROOT / "std.out"
IMAGE_DPI = 800
ENCODING = "utf-8"


def get_divisors(n, minimum=5):
    bin_nums = np.arange(minimum, n + 1)
    remainders = np.remainder(n, bin_nums)
    mask = remainders == 0
    return bin_nums[mask].tolist()
