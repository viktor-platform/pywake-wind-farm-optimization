import re

import numpy as np
import requests
import xarray as xr


def get_gwc_data(
    latitude: float, longitude: float, encoding: str = "ASCII"
) -> xr.Dataset:
    """
    Gets GWC data at gievn coordinates. Source: https://github.com/jules-ch/wind-stats Copyright (c) 2020 Jules Chéron (MIT)
    """

    # global windatlas API
    gwc_bytes = requests.get(
        f"https://globalwindatlas.info/api/gwa/custom/Lib/?lat={latitude}&long={longitude}",
        headers={"Referer": "https://globalwindatlas.info"},
    ).content  # byte array with gwc content

    gwc = gwc_bytes.decode(encoding)
    pattern = "<coordinates>(.*)</coordinates>"

    lines = gwc.splitlines()
    roughness_classes, heights_count, sectors_count = map(int, lines[1].split())
    sectors = [360 / sectors_count * i for i in range(sectors_count)]

    coordinates_match = re.search(pattern, gwc)

    if coordinates_match:
        # longitude, latitude which is not consistent with the documentation somehow
        longitude, latitude, _ = map(float, coordinates_match.group(1).split(","))
    else:
        raise ValueError("coordinates not found in the GWC file")  # pragma: no cover
    coordinates = (latitude, longitude)
    roughness_lengths = list(map(float, lines[2].split()))
    heights = list(map(float, lines[3].split()))

    weibull_data_lines = lines[4:]
    data_array = np.asarray([line.split() for line in weibull_data_lines], float)

    A_weibull = np.zeros((roughness_classes, heights_count, sectors_count), float)
    k_weibull = np.zeros((roughness_classes, heights_count, sectors_count), float)
    frequencies = np.zeros((roughness_classes, sectors_count), float)
    for roughness_index in range(roughness_classes):
        index = roughness_index * (heights_count * 2 + 1)
        roughness_data = data_array[index + 1 : index + heights_count * 2 + 1]
        frequencies[roughness_index] = data_array[index]

        for height_index in range(heights_count):
            A, K = roughness_data[height_index * 2 : (height_index * 2) + 2]
            A_weibull[roughness_index][height_index] = A
            k_weibull[roughness_index][height_index] = K

    return xr.Dataset(
        {
            "A": (["roughness", "height", "sector"], A_weibull),
            "k": (["roughness", "height", "sector"], k_weibull),
            "frequency": (["roughness", "sector"], frequencies),
        },
        coords={
            "roughness": roughness_lengths,
            "height": heights,
            "sector": sectors,
        },
        attrs={"coordinates": coordinates},
    )
