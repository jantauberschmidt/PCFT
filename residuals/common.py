"""
Shared utilities for patch-based weak residual estimators.

Mathematical specification
--------------------------
For a scalar field ``u`` sampled on a Cartesian grid, this module builds test
functions of the form

``psi_n(x) = phi_n(x) * m(x)``,

where:
- ``phi_n`` is a compactly supported Wendland-type radial basis function
  centered at a random location with random per-axis scale ``sigma``,
- ``m(x) = prod_i ((x_i-a_i)(b_i-x_i)/(b_i-a_i)^2)`` is a bridge mollifier
  vanishing on the domain boundary.

For each test center, the code extracts a local ``K^d`` patch (``K=2*ceil(sigma_max)+1``),
applies trapezoidal quadrature weights, and normalizes ``psi_n`` by its
``H^1`` seminorm:

``||psi_n||_{H^1_0} ~= sqrt(sum_grid |grad psi_n|^2 w dV)``.

Inputs/Outputs
--------------
- Functions operate on ``torch.Tensor`` data on CPU/GPU.
- Spatial derivative convention from ``torch.gradient`` is axis order
  ``(y, x)`` in 2-D and extends naturally to ``d`` dimensions.
- ``prepare_test_functions_nd`` returns patch tensors used by PDE residual
  modules:
  ``(u_patch, a_patch, source_patch, grad_u_patch, phi_moll, grad_phi_moll,
  center_coords, weights_patch, dV)``.

Autograd and stochasticity
--------------------------
- Test-function construction is wrapped in ``torch.no_grad()``.
- ``grad_u_patch`` is computed from ``u`` outside ``no_grad``, so gradients can
  propagate to ``u`` through weak residual assembly.
- Random sampling uses global PyTorch RNG via ``torch.rand``/``torch.randn``.
"""

import torch
import torch.nn.functional as F
import math


def generate_grid_nd(axes, ranges, device=None):
    """
    Create an ``n``-D coordinate grid over axis-aligned ranges.

    Args:
        axes: Sequence of grid sizes ``(N_1, ..., N_d)``.
        ranges: Sequence of coordinate ranges ``[(a_1,b_1), ..., (a_d,b_d)]``.
        device: Optional target device.

    Returns:
        Tensor of shape ``(N_1, ..., N_d, d)`` with coordinate components in
        the last dimension.
    """
    assert len(axes) == len(ranges), "Length of axes and ranges must match."
    dim = len(axes)

    grids = [
        torch.linspace(ranges[i][0], ranges[i][1], steps=axes[i], device=device)
        for i in range(dim)
    ]

    mesh = torch.meshgrid(*grids, indexing='ij')  # list of [N₁, ..., N_d] tensors
    coords = torch.stack(mesh, dim=-1)  # shape: [N₁, ..., N_d, d]

    return coords


def extract_patches_nd(grid, idx_list, batched=False):
    """Gather local patches from an ``n``-D tensor using precomputed indices.

    Parameters
    ----------
    grid : torch.Tensor
        Either scalar-valued ``(...spatial...)`` or vector-valued
        ``(...spatial..., C)``; if ``batched=True``, includes leading batch dim.
    idx_list : list[torch.Tensor]
        One integer index tensor per spatial axis, each of shape
        ``(N_test, K_1, ..., K_d)``.
    batched : bool
        If true, treats first tensor dimension as batch and extracts patches for
        every batch element.

    Returns
    -------
    torch.Tensor
        Patch tensor with shape:
        - unbatched scalar: ``(N_test, K_1, ..., K_d)``,
        - unbatched vector: ``(N_test, K_1, ..., K_d, C)``,
        - batched scalar: ``(B, N_test, K_1, ..., K_d)``,
        - batched vector: ``(B, N_test, K_1, ..., K_d, C)``.
    """
    ndim = len(idx_list)
    shape = idx_list[0].shape
    N = shape[0]
    patch_shape = shape[1:]

    if not batched:
        idx_flat = [ix.reshape(-1) for ix in idx_list]

        if grid.ndim == ndim + 1:
            C = grid.shape[-1]
            gathered = grid[tuple(idx_flat)].view(*shape, C)
        elif grid.ndim == ndim:
            gathered = grid[tuple(idx_flat)].view(*shape)
        else:
            raise ValueError(f"Unexpected grid.ndim = {grid.ndim}")
        return gathered

    B = grid.shape[0]
    idx_flat = [ix.reshape(N, -1) for ix in idx_list]
    num_points = idx_flat[0].shape[1]

    batch_idx = torch.arange(B, device=grid.device).view(B, 1, 1)
    batch_idx = batch_idx.expand(B, N, num_points).reshape(-1)

    idx_expanded = [ix.unsqueeze(0).expand(B, -1, -1) for ix in idx_flat]
    idx_expanded = [ix.reshape(-1) for ix in idx_expanded]

    if grid.ndim == ndim + 2:
        C = grid.shape[-1]
        gathered = grid[batch_idx, *idx_expanded]
        return gathered.view(B, N, *patch_shape, C)
    elif grid.ndim == ndim + 1:
        gathered = grid[batch_idx, *idx_expanded]
        return gathered.view(B, N, *patch_shape)
    else:
        raise ValueError(f"Unexpected grid.ndim = {grid.ndim} for batched=True")

def bridge_mollifier_1d(x, a, b):
    """Return 1-D bridge mollifier ``m(x)=((x-a)(b-x))/(b-a)^2``."""
    return ((x - a) * (b - x)) / (b - a) ** 2


def grad_bridge_mollifier_1d(x, a, b):
    """Return derivative ``dm/dx = (a+b-2x)/(b-a)^2`` of bridge mollifier."""
    return (a + b - 2 * x) / (b - a) ** 2

def compute_mollifier_nd(coords, ranges):
    """Build separable ``n``-D bridge mollifier and its gradient.

    Parameters
    ----------
    coords : torch.Tensor
        Coordinate grid ``(..., d)``.
    ranges : sequence[tuple[float, float]]
        Domain bounds for each coordinate axis.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``mollifier`` with shape ``(...,)`` and ``grad_mollifier`` with shape
        ``(..., d)``.
    """
    ndim = coords.shape[-1]

    ms = []
    dms = []

    for i in range(ndim):
        m_i = bridge_mollifier_1d(coords[..., i], *ranges[i])
        dm_i = grad_bridge_mollifier_1d(coords[..., i], *ranges[i])
        ms.append(m_i)
        dms.append(dm_i)

    mollifier = torch.prod(torch.stack(ms, dim=-1), dim=-1)

    grad_m = []
    for i in range(ndim):
        other_m = torch.prod(torch.stack([ms[j] for j in range(ndim) if j != i], dim=-1), dim=-1)
        grad_m.append(dms[i] * other_m)

    grad_mollifier = torch.stack(grad_m, dim=-1)

    return mollifier, grad_mollifier

def trapezoidal_weights_nd(shape, device=None):
    """
    Create nD trapezoidal integration weights.

    Args:
        shape:    list or tuple of ints, grid size in each dimension
        device:   torch device (optional)
    Returns:
        weights:  ``torch.Tensor`` of shape ``shape`` with separable trapezoidal
                  factors (0.5 on boundary indices per axis).
    """
    ndim = len(shape)
    weights = torch.ones(*shape, device=device)

    for dim in range(ndim):
        # Build slicing tuple to select the first and last index along `dim`
        first = [slice(None)] * ndim
        last = [slice(None)] * ndim
        first[dim] = 0
        last[dim] = -1

        weights[tuple(first)] *= 0.5
        weights[tuple(last)] *= 0.5

    return weights


def pad_spatial_nd(x, pad_width, is_vector=False, is_batched=True, value=0.0):
    """
    Pad all spatial dimensions equally for scalar or vector-valued tensors.

    Args:
        x: Tensor of shape
            - [B, d1, ..., dn] or [B, d1, ..., dn, D] if is_batched
            - [d1, ..., dn] or [d1, ..., dn, D] if not batched
        pad_width: int, padding size per spatial dimension (same left/right)
        is_vector: if True, the last dim is not spatial (e.g. vector/component dim)
        is_batched: if True, the first dimension is batch
        value: constant value to pad with

    Returns:
        Tensor with same shape layout as input and spatial dimensions padded by
        ``pad_width`` on both sides.
    """
    mode = 'constant'
    ndim = x.ndim

    n_spatial = ndim - 2 if (is_batched and is_vector) else \
                ndim - 1 if (is_batched and not is_vector) else \
                ndim - 1 if (not is_batched and is_vector) else \
                ndim

    # Create reversed pad tuple: [dim_n_left, dim_n_right, ..., dim_1_left, dim_1_right]
    pad_tuple = [pad_width, pad_width] * n_spatial
    pad_tuple = tuple(reversed(pad_tuple))

    # Determine permutation if vector
    if is_vector:
        if is_batched:
            # x: [B, d1, ..., dn, D] → [B, D, d1, ..., dn]
            x_perm = x.movedim(-1, 1)
            x_pad = F.pad(x_perm, pad=pad_tuple, mode=mode, value=value)
            x_final = x_pad.movedim(1, -1)
        else:
            # x: [d1, ..., dn, D] → [D, d1, ..., dn]
            x_perm = x.movedim(-1, 0)
            x_pad = F.pad(x_perm, pad=pad_tuple, mode=mode, value=value)
            x_final = x_pad.movedim(0, -1)
    else:
        x_final = F.pad(x, pad=pad_tuple, mode=mode, value=value)

    return x_final



def compute_wendland_test_function_nd(axes, ranges, sigma_range, max_sigma, coords, coords_pad, dxs, device, N_test=None):
    """
    Sample Wendland-C2 radial test functions and gradients on extracted patches.

    Radial profile:
    ``phi(r) = (1-r)_+^4 (4r+1)``, where
    ``r = || (x-c) / (sigma * dx) ||_2`` with per-axis random ``sigma``.

    Randomness:
    - Centers and scales use global PyTorch RNG.
    - If ``N_test is None``, one center per grid point is used with small random
      jitter and clamped to the range.

    Returns:
        phi:      [N, K, ..., K]
        grad_phi: [N, K, ..., K, ndim]
        center_coords: list of center coordinates per axis [N]
        patch_indices: list of [N, K, ..., K] index tensors (per axis)
    """
    ndim = len(axes)
    K = 2 * max_sigma + 1
    grid = torch.arange(K, device=device)
    dx_tensor = torch.tensor(dxs).to(device)

    if N_test is None:
        N_test = math.prod(axes)
        # Generate centers directly in the range for each dimension
        center_coords = coords + 0.1 * dx_tensor * torch.randn_like(coords, device=device)
        center_coords = torch.clamp(center_coords, min=ranges[0][0], max=ranges[0][1])  # Ensure within bounds
        center_coords = [center_coords[..., i].reshape(-1) for i in range(ndim)]
    else:
        # Sample centers randomly within the given ranges
        center_coords = [((ranges[i][1] - ranges[i][0]) * torch.rand(N_test, device=device) + ranges[i][0])
                         for i in range(ndim)]

    sigma_coords = [torch.rand(N_test, device=device) * (sigma_range[1] - sigma_range[0]) + sigma_range[0]
                    for _ in range(ndim)]

    # Build meshgrid for patch offsets (we do this only once)
    mesh = torch.meshgrid([grid] * ndim, indexing='ij')  # list of [K, K, ...]
    offsets = torch.stack(mesh, dim=-1).view(1, *([K] * ndim), ndim)  # Shape: [1, K, K, ..., ndim]

    # Efficiently compute the normalized indices for each center coordinate
    norm_center_coords = [(center_coords[d] - ranges[d][0]) / (ranges[d][1] - ranges[d][0]) for d in range(ndim)]
    center_idx = [torch.round(norm * (axes[d] - 1)).long() + max_sigma for d, norm in enumerate(norm_center_coords)]
    start_idx = [idx - max_sigma for idx in center_idx]

    # Generate indices for each axis at once
    idx = [start.view(-1, *[1] * ndim) + offsets[..., d] for d, start in enumerate(start_idx)]

    # Extract patches
    coords_patch = extract_patches_nd(coords_pad, idx)

    # Compute r
    sigma_stack = torch.stack(sigma_coords, dim=-1).view(N_test, *[1] * ndim, ndim)  # [N, 1,...,1, ndim]
    center_stack = torch.stack(center_coords, dim=-1).view(N_test, *[1] * ndim, ndim)
    diff = (coords_patch - center_stack) / (sigma_stack * dx_tensor.view(1, *[1]*ndim, ndim))  # [N, ..., ndim]

    r = torch.norm(diff, dim=-1)  # [N, K, ..., K]
    one_minus_r = (1.0 - r).clamp(min=0.)
    phi = one_minus_r ** 4 * (4 * r + 1)  # [N, ..., K]
    dphi_dr = torch.where(r > 1e-8, -16 * r * one_minus_r ** 3, torch.zeros_like(r))

    unit_scaled_diff = torch.where(
        r[..., None] > 1e-8,
        diff / r[..., None],
        torch.zeros_like(diff)
    )  # [N, ..., ndim]

    grad_r = unit_scaled_diff / (sigma_stack * dx_tensor.view(1, *[1]*ndim, ndim))
    grad_phi = dphi_dr[..., None] * grad_r  # [N, ..., ndim]

    return phi, grad_phi, center_coords, idx

def compute_wendland_wavelet_function_nd(axes, ranges, sigma_range, max_sigma, coords, coords_pad, dxs, device, N_test=None):
    """
    Sample a randomized Wendland-like wavelet family and gradients.

    Uses the same base ``phi_w(r) = (1-r)_+^4 (4r+1)`` as
    ``compute_wendland_test_function_nd`` multiplied by
    ``(1 - b * 64 * r^4)``, where ``b in {0,1}`` is Bernoulli(0.5) drawn per
    test function. This randomness changes both amplitude and derivative.

    Returns:
        phi:      [N, K, ..., K]
        grad_phi: [N, K, ..., K, ndim]
        center_coords: list of center coordinates per axis [N]
        patch_indices: list of [N, K, ..., K] index tensors (per axis)
    """
    ndim = len(axes)
    K = 2 * max_sigma + 1
    grid = torch.arange(K, device=device)
    dx_tensor = torch.tensor(dxs).to(device)

    if N_test is None:
        N_test = math.prod(axes)
        # Generate centers directly in the range for each dimension
        center_coords = coords + 0.1 * dx_tensor * torch.randn_like(coords, device=device)
        center_coords = torch.clamp(center_coords, min=ranges[0][0], max=ranges[0][1])  # Ensure within bounds
        center_coords = [center_coords[..., i].reshape(-1) for i in range(ndim)]
    else:
        # Sample centers randomly within the given ranges
        center_coords = [((ranges[i][1] - ranges[i][0]) * torch.rand(N_test, device=device) + ranges[i][0])
                         for i in range(ndim)]

    sigma_coords = [torch.rand(N_test, device=device) * (sigma_range[1] - sigma_range[0]) + sigma_range[0]
                    for _ in range(ndim)]

    # Build meshgrid for patch offsets (we do this only once)
    mesh = torch.meshgrid([grid] * ndim, indexing='ij')  # list of [K, K, ...]
    offsets = torch.stack(mesh, dim=-1).view(1, *([K] * ndim), ndim)  # Shape: [1, K, K, ..., ndim]

    # Efficiently compute the normalized indices for each center coordinate
    norm_center_coords = [(center_coords[d] - ranges[d][0]) / (ranges[d][1] - ranges[d][0]) for d in range(ndim)]
    center_idx = [torch.round(norm * (axes[d] - 1)).long() + max_sigma for d, norm in enumerate(norm_center_coords)]
    start_idx = [idx - max_sigma for idx in center_idx]

    # Generate indices for each axis at once
    idx = [start.view(-1, *[1] * ndim) + offsets[..., d] for d, start in enumerate(start_idx)]

    # Extract patches
    coords_patch = extract_patches_nd(coords_pad, idx)

    # Compute r
    sigma_stack = torch.stack(sigma_coords, dim=-1).view(N_test, *[1] * ndim, ndim)  # [N, 1,...,1, ndim]
    center_stack = torch.stack(center_coords, dim=-1).view(N_test, *[1] * ndim, ndim)
    diff = (coords_patch - center_stack) / (sigma_stack * dx_tensor.view(1, *[1]*ndim, ndim))  # [N, ..., ndim]

    r = torch.norm(diff, dim=-1)  # [N, K, ..., K]
    one_minus_r = (1.0 - r).clamp(min=0.)
    b = (torch.randn(N_test) > 0).float().view(-1, *[1] * ndim).to(device)
    phi = one_minus_r ** 4 * (4 * r + 1)  * (1 - b * 64 * r**4)
    # [N, ..., K]
    dphi_dr_raw = (-4 * one_minus_r**3 * (4*r + 1) * (1 - b * 64 * r **4)
                   + 4 * one_minus_r**4 * (1 - b * 64 * r**4)
                   - b * 64 * 4 * r**3 * one_minus_r**4 * (4*r + 1))
    dphi_dr = torch.where(r > 1e-8, dphi_dr_raw, torch.zeros_like(r))

    unit_scaled_diff = torch.where(
        r[..., None] > 1e-8,
        diff / r[..., None],
        torch.zeros_like(diff)
    )  # [N, ..., ndim]

    grad_r = unit_scaled_diff / (sigma_stack * dx_tensor.view(1, *[1]*ndim, ndim))
    grad_phi = dphi_dr[..., None] * grad_r  # [N, ..., ndim]

    return phi, grad_phi, center_coords, idx


def prepare_test_functions_nd(u, a, ranges, source=None, device='cpu', sigma_range=(5., 20.), N_test=None, test_fun='wd_wv'):
    """Prepare local test-function patches and field patches for weak residuals.

    Parameters
    ----------
    u : torch.Tensor
        Scalar field batch of shape ``(B, N_1, ..., N_d)``.
    a : torch.Tensor
        Coefficient field batch, same shape as ``u``.
    ranges : sequence[tuple[float, float]]
        Physical coordinate ranges per axis.
    source : torch.Tensor or None
        Optional source field. Supported shapes:
        ``(N_1,...,N_d)``, ``(B,N_1,...,N_d)``, or multi-channel
        ``(C,N_1,...,N_d)``.
    device : str or torch.device
        Device used for generated coordinates/test functions.
    sigma_range : tuple[float, float]
        Min/max support-radius parameter passed to test-function samplers.
    N_test : int or None
        Number of test functions. If ``None``, defaults to product of grid sizes.
    test_fun : str
        ``"wd_wv"`` for randomized wavelet variant, ``"wd"`` for standard
        Wendland-C2.

    Returns
    -------
    tuple
        ``(u_patch, a_patch, source_patch, grad_u_patch, phi_moll,
        grad_phi_moll, center_coords, weights_patch, dA)`` with:
        - ``u_patch``: ``(B,N_test,K_1,...,K_d)``,
        - ``a_patch``: same as ``u_patch``,
        - ``source_patch``: ``None`` or source patches with matching local axes,
        - ``grad_u_patch``: ``(B,N_test,K_1,...,K_d,d)``,
        - ``phi_moll``: ``(N_test,K_1,...,K_d)``,
        - ``grad_phi_moll``: ``(N_test,K_1,...,K_d,d)``,
        - ``center_coords``: list of length ``d`` with center coordinates,
        - ``weights_patch``: trapezoidal weights on each patch,
        - ``dA``: scalar grid cell/voxel measure ``prod_i dx_i``.
    """

    B, *axes = u.shape
    spatial_dims = tuple(range(1, u.ndim))
    dxs = [(r[1] - r[0]) / (axes[i] - 1) for (i, r) in enumerate(ranges)]

    dA = math.prod(dxs)
    max_sigma = math.ceil(sigma_range[1])

    grad_u = torch.gradient(u, dim=spatial_dims, spacing=dxs, edge_order=2)
    grad_u = torch.stack(grad_u, dim=-1)

    a_pad = pad_spatial_nd(a, max_sigma, is_vector=False)
    u_pad = pad_spatial_nd(u, max_sigma, is_vector=False)
    grad_u_pad = pad_spatial_nd(grad_u, max_sigma, is_vector=True)

    with torch.no_grad():
        coords = generate_grid_nd(axes, ranges, device=device)
        weights = trapezoidal_weights_nd(axes, device=device)
        mollifier, grad_mollifier = compute_mollifier_nd(coords, ranges)



        weights_pad = pad_spatial_nd(weights, max_sigma, is_vector=False, is_batched=False)
        mollifier_pad = pad_spatial_nd(mollifier, max_sigma, is_vector=False, is_batched=False)

        grad_mollifier_pad = pad_spatial_nd(grad_mollifier, max_sigma, is_vector=True, is_batched=False)
        coords_pad = pad_spatial_nd(coords, max_sigma, is_vector=True, is_batched=False)


        if test_fun == 'wd_wv':
            phi, grad_phi, center_coords, idx = compute_wendland_wavelet_function_nd(axes=axes,
                                                                                     ranges=ranges,
                                                                                     sigma_range=sigma_range,
                                                                                     max_sigma=max_sigma,
                                                                                     coords=coords,
                                                                                     coords_pad=coords_pad,
                                                                                     dxs=dxs,
                                                                                     device=device,
                                                                                     N_test=N_test)

        elif test_fun == 'wd':
            phi, grad_phi, center_coords, idx = compute_wendland_test_function_nd(axes=axes,
                                                                                     ranges=ranges,
                                                                                     sigma_range=sigma_range,
                                                                                     max_sigma=max_sigma,
                                                                                     coords=coords,
                                                                                     coords_pad=coords_pad,
                                                                                     dxs=dxs,
                                                                                     device=device,
                                                                                     N_test=N_test)

        else:
            raise ValueError('Unknown specification of test function.')




        weights_patch = extract_patches_nd(weights_pad, idx)  # [N, K, ..., K]
        mollifier_patch = extract_patches_nd(mollifier_pad, idx)  # [N, K, ..., K]
        grad_mollifier_patch = extract_patches_nd(grad_mollifier_pad, idx)



        phi_moll = phi * mollifier_patch
        grad_phi_moll = grad_phi * mollifier_patch[..., None] + \
                        phi[..., None] * grad_mollifier_patch


        # normalise test functions in operator norm. Use semi-norm (gradient only).
        norm_grad_sq = (grad_phi_moll ** 2).sum(-1) * weights_patch * dA  # Integrate ||∇ϕ||²
        norm = torch.sqrt(norm_grad_sq.sum(dim=spatial_dims)) + 1e-10

        grad_phi_moll = grad_phi_moll / norm.view([-1,] + [1] * (len(spatial_dims) + 1))
        phi_moll = phi_moll / norm.view([-1,] + [1] * len(spatial_dims))


    a_patch = extract_patches_nd(a_pad, idx, batched=True)
    u_patch = extract_patches_nd(u_pad, idx, batched=True)
    grad_u_patch = extract_patches_nd(grad_u_pad, idx, batched=True)

    if source is None:
        source_patch = None
    else:
        if source.ndim == 2:
            source_pad = pad_spatial_nd(source, max_sigma, is_vector=False, is_batched=False)
            source_patch = extract_patches_nd(source_pad, idx, batched=False)
        else:
            source_pad = pad_spatial_nd(source, max_sigma, is_vector=False, is_batched=True)
            source_patch = extract_patches_nd(source_pad, idx, batched=True)

    return u_patch, a_patch, source_patch, grad_u_patch, phi_moll, grad_phi_moll, center_coords, weights_patch, dA
