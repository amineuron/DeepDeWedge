import math

import torch

from .fourier import get_3d_fft_freqs_on_grid
from .rotation import rotate_vol_around_axis


# The missing-wedge mask is fully determined by (grid_size, mw_angle, device) and
# is read-only downstream, but building it costs ~0.5 s for a 96^3 grid (and more
# for the sqrt(2)-enlarged grid used by the rotated mask) and was previously rebuilt
# for EVERY dataset item, every epoch. Memoize it -> identical output, big speedup.
_MW_MASK_CACHE = {}


def get_missing_wedge_mask(grid_size, mw_angle, device="cpu"):
    """
    Produces a 3D binary mask with shape 'grid_size', which can be used to zero-out Fourier components that lie inside a missing wedge with width 'mw_angle'. Memoized (see _MW_MASK_CACHE): the result is deterministic in its arguments and used read-only.
    """
    _key = (tuple(int(s) for s in grid_size), float(mw_angle), str(device))
    _cached = _MW_MASK_CACHE.get(_key)
    if _cached is not None:
        return _cached
    grid = get_3d_fft_freqs_on_grid(grid_size=grid_size, device=device)
    # make normal vectors of two hyperplanes that bound missing wedge
    alpha = torch.deg2rad(torch.tensor(float(mw_angle))) / 2
    normal_left = torch.tensor([torch.sin(alpha), torch.cos(alpha)])
    normal_right = torch.tensor([torch.sin(alpha), -torch.cos(alpha)])
    # embed normal vectors into x-z plane
    normal_left = torch.tensor([normal_left[0], 0, normal_left[1]], device=device)
    normal_right = torch.tensor([normal_right[0], 0, normal_right[1]], device=device)
    # select all points that lie above or below both hyperplanes that bound missing wedge
    # convert to list because reshape needs list or tuple
    grid_size = [int(s) for s in grid_size]
    upper_wedge = torch.logical_or(
        grid.inner(normal_left) >= 0, grid.inner(normal_right) >= 0
    ).reshape(list(grid_size))
    lower_wedge = torch.logical_or(
        grid.inner(normal_left) <= 0, grid.inner(normal_right) <= 0
    ).reshape(list(grid_size))
    mw_mask = torch.logical_and(upper_wedge, lower_wedge).int()
    _MW_MASK_CACHE[_key] = mw_mask
    return mw_mask


def get_rotated_missing_wedge_mask(
    grid_size, mw_angle, rot_axis, rot_angle, device="cpu"
):
    """
    Convenience function that generates a missing wedge mask and rotates it 'rot_angle' degrees around 'rot_axis'.
    """
    grid_size = torch.tensor(grid_size)
    # enlarge grid size such that rotated grid fits inside
    adjusted_grid_size = (torch.ceil(math.sqrt(2) * grid_size) / 2.0) * 2
    mw_mask = get_missing_wedge_mask(grid_size=adjusted_grid_size, mw_angle=mw_angle)
    mw_mask = (
        rotate_vol_around_axis(
            vol=mw_mask,
            rot_angle=rot_angle,
            rot_axis=rot_axis,
            output_shape=grid_size,
            order=3,
        )
        .float()
        .to(device)
    )
    return mw_mask
