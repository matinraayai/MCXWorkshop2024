from typing import *
import numpy as np
from numpy.random import rand
from .volume import random_volume, colin27_volume, usc195_volume, digimouse_volume


def random_train_cfg(vol_size: Union[Tuple[int, int], Tuple[int, int, int]],
                     num_props: int,
                     gpu_ids: str) -> Dict[str, Any]:
    """
    Generates random configuration for running a 2D/3D simulation in the MCX software.
    The output doesn't have a nphoton attribute which should be set manually before simulation
    :param nphoton: Number of simulation photons
    :param vol_size: Size of the simulation volumes
    :param num_props: Number of props placed in the volume
    :param gpu_ids: ID of the GPUs used for simulation
    :return:
    """
    vol = random_volume(vol_size, num_props)
    vol += 1
    max_prop = vol.max(initial=None)
    # Add a third axis to the volume. MCX doesn't accept 2D volumes directly.
    cfg = {
        "vol": vol,
        "issrcfrom0": 1,  # Disables legacy volume indexing
        "srctype": 'isotropic',
        "gpuid": gpu_ids,
        "autopilot": 1,
        "seed": np.random.randint(2 ** 31 - 1)
    }
    if len(vol_size) == 2:
        cfg["srcpos"] = (0, rand() * vol_size[0], rand() * vol_size[1])
        cfg["srcdir"] = [0,
                         vol_size[0] * 0.5 - cfg["srcpos"][0],
                         vol_size[1] * 0.5 - cfg["srcpos"][1]]
    else:
        cfg["srcpos"] = (rand() * vol_size[0], rand() * vol_size[1], rand() * vol_size[2])
        cfg["srcdir"] = [vol_size[0] * 0.5 - cfg["srcpos"][0],
                         vol_size[1] * 0.5 - cfg["srcpos"][1],
                         vol_size[2] * 0.5 - cfg["srcpos"][2]]
    cfg["srcdir"] /= np.linalg.norm(cfg["srcdir"])
    # Property generation
    g = 0.1 * np.random.rand(max_prop, 1) + 0.9
    musp = np.abs(np.random.randn(max_prop, 1))
    mus = musp / (1 - g)
    props = np.hstack((
        np.abs(np.random.randn(max_prop, 1) * 0.05 + 0.01),
        mus,
        g,
        9 * np.random.rand(max_prop, 1) + 1
    ))
    cfg["prop"] = np.vstack(([0, 0, 1, 1], props))
    # Time setting
    cfg["tstart"] = 0
    cfg["tend"] = 1e-9 * np.random.randint(5, 15)
    cfg["tstep"] = cfg["tend"]

    return cfg


def colin27_cfg(nphoton: int, gpu_ids: str) -> Dict[str, Any]:
    return {
        "nphoton": nphoton,
        "vol": colin27_volume(),
        "srcpos": [75, 67.38, 167.5],
        "srcdir": [0.16360629, 0.45691758, -0.87433364],
        "prop": [[0, 0, 1.0000, 1.0000],  # background / air
                 [0.0190, 7.8182, 0.8900, 1.3700],  # scalp
                 [0.0190, 7.8182, 0.8900, 1.3700],  # skull
                 [0.0040, 0.0090, 0.8900, 1.3700],  # csf
                 [0.0200, 9.0000, 0.8900, 1.3700],  # gray matters
                 [0.0800, 40.9000, 0.8400, 1.3700],  # white matters
                 [0, 0, 1.0000, 1.0000]],  # air pockets
        "issrcfrom0": 1,
        "seed": np.random.randint(2 ** 32 - 1),
        "gpuid": gpu_ids,
        "tstart": 0,
        "tend": 1e-8,
        "tstep": 1e-8
    }


def digimouse_cfg(nphoton: int, gpu_ids: str) -> Dict[str, Any]:
    return {
        "nphoton": nphoton,
        "vol": digimouse_volume(),
        "prop": [[0, 0, 1.0000, 1.0000],
                 [0.0191, 6.6000, 0.9000, 1.3700],
                 [0.0136, 8.6000, 0.9000, 1.3700],
                 [0.0026, 0.0100, 0.9000, 1.3700],
                 [0.0186, 11.1000, 0.9000, 1.3700],
                 [0.0186, 11.1000, 0.9000, 1.3700],
                 [0.0186, 11.1000, 0.9000, 1.3700],
                 [0.0186, 11.1000, 0.9000, 1.3700],
                 [0.0186, 11.1000, 0.9000, 1.3700],
                 [0.0240, 8.9000, 0.9000, 1.3700],
                 [0.0026, 0.0100, 0.9000, 1.3700],
                 [0.0240, 8.9000, 0.9000, 1.3700],
                 [0.0240, 8.9000, 0.9000, 1.3700],
                 [0.0240, 8.9000, 0.9000, 1.3700],
                 [0.0240, 8.9000, 0.9000, 1.3700],
                 [0.0240, 8.9000, 0.9000, 1.3700],
                 [0.0720, 5.6000, 0.9000, 1.3700],
                 [0.0720, 5.6000, 0.9000, 1.3700],
                 [0.0720, 5.6000, 0.9000, 1.3700],
                 [0.0500, 5.4000, 0.9000, 1.3700],
                 [0.0240, 8.9000, 0.9000, 1.3700],
                 [0.0760, 10.9000, 0.9000, 1.3700]],
        "srctype": 'fourier',
        "srcpos": [50.0, 200.0, 100.0],
        "srcparam1": [100.0, 0.0, 0.0, 2],
        "srcparam2": [0, 100.0, 0.0, 0],
        "srcdir": [0, 0, -1],
        "issrcfrom0": 1,
        "tstart": 0,
        "tend": 5e-9,
        "tstep": 5e-9,
        "autopilot": 1,
        "gpuid": gpu_ids,
        "unitinmm": 0.8,
        "seed": np.random.randint(2**32 - 1)
    }


def usc195_cfg(nphoton: int, gpu_ids: str) -> Dict[str, Any]:
    return {
        "nphoton": nphoton,
        "vol": usc195_volume(),
        "prop": [[0, 0, 1, 1], # Ambient air
                 [0.019, 7.8, 0.89, 1.37], # Scalp
                 [0.02, 9.0, 0.89, 1.37], # Skull
                 [0.004, 0.009, 0.89, 1.37], # CSF
                 [0.019, 7.8, 0.89, 1.37], # Gray matter
                 [0.08, 40.9, 0.84, 1.37], # White matter
                 [0, 0, 1, 1]], # Air cavities
        "srcnum": 1,
        "srcpos": [133.5370, 90.1988, 200.0700], # pencil beam source placed at EEG 10-5 landmark:"C4h"
        "srctype": 'pencil',
        "srcdir": [-0.5086, -0.1822, -0.8415], # inward-pointing source
        "issrcfrom0": 1,
        "tstart": 0,
        "tend": 1e-8,
        "tstep": 1e-8,
        "isspecular": 0,
        "autopilot": 1,
        "gpuid": gpu_ids,
        "seed": np.random.randint(2**32 - 1)
    }


def cube_benchmark_cfg(cube_type: str, nphoton: int, gpu_ids: str,
                       vol_dims: Union[Tuple[int, int], Tuple[int, int, int]]) -> Dict[str, Any]:
    cfg = {
        "nphoton": nphoton,
        "issrcfrom0": 1,
        "srcdir": [0, 0, 1],
        "gpuid": gpu_ids,
        "autopilot": 1,
        "seed": np.random.randint(2**31 - 1),
        "tstart": 0,
        "tend": 1e-8,
        "tstep": 1e-8
    }
    if len(vol_dims) == 2:
        cfg["srcpos"] = [0, vol_dims[1] // 2, 0]
        cfg["vol"] = np.expand_dims(np.ones(vol_dims, dtype=np.uint8), 0)
        cfg["vol"][:, (vol_dims[0] * 3) // 10: (vol_dims[0] * 7) // 10, vol_dims[1] // 10: (5 * vol_dims[1]) // 10] = 2
    else:
        cfg["srcpos"] = [vol_dims[0] // 2, vol_dims[1] // 2, 0]
        cfg["vol"] = np.ones(vol_dims, dtype=np.uint8)
        cfg["vol"][(vol_dims[0] * 3) // 10: (vol_dims[0] * 7) // 10,
                   (vol_dims[1] * 3) // 10: (vol_dims[1] * 7) // 10,
                   vol_dims[2] // 10: vol_dims[2] // 2
                   ] = 2

    if cube_type.lower() == "homo":
        cfg["prop"] = [[0, 0, 1, 1],
                       [0.02, 10, 0.9, 1.37],
                       [0.02, 10, 0.9, 1.37]]
    elif cube_type.lower() == "absorb":
        cfg["prop"] = [[0, 0, 1, 1],
                       [0.02, 10, 0.9, 1.37],
                       [0.1, 10, 0.9, 1.37]]
    elif cube_type.lower() == "refractive":
        cfg["prop"] = [[0, 0, 1, 1],
                       [0.02, 10, 0.9, 1.37],
                       [0.02, 10, 0.9, 6.85]]
    return cfg


