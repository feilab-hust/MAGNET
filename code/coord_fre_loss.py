"""Frequency-selective loss for coordinate-query grid artifacts."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _alias_frequency(frequency: float) -> float:
    """Fold a frequency in cycles/pixel into the sampled [0, 0.5] band."""
    wrapped = frequency % 1.0
    return min(wrapped, 1.0 - wrapped)


def _periodic_distance(axis: torch.Tensor, center: float) -> torch.Tensor:
    distance = (axis - center).abs()
    return torch.minimum(distance, 1.0 - distance)


def _carrier_centers(frequency: float, include_axes: bool,
                     include_diagonals: bool) -> list[tuple[float, float]]:
    signs = (-1.0, 1.0)
    centers: list[tuple[float, float]] = []
    if include_axes:
        centers.extend((sign * frequency, 0.0) for sign in signs)
        centers.extend((0.0, sign * frequency) for sign in signs)
    if include_diagonals:
        centers.extend(
            (sign_y * frequency, sign_x * frequency)
            for sign_y in signs for sign_x in signs
        )
    return centers


def coordinate_frequency_loss(
    image: torch.Tensor,
    scale_factor: float,
    band_width: float = 0.035,
    ring_inner_ratio: float = 1.5,
    ring_outer_ratio: float = 3.0,
    peak_ratio: float = 1.25,
    harmonics: int = 1,
    include_axes: bool = True,
    include_diagonals: bool = True,
    min_frequency: float = 0.08,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Penalize narrow spectral peaks tied to a coordinate-query lattice.

    A query scale ``s`` places the input-pixel lattice carrier at ``1 / s``
    cycles per output pixel. Sampling folds that carrier into [0, 0.5]. For
    every selected carrier, its spectral power is compared with a surrounding
    ring. Broadband biological edges raise both regions, while a coordinate
    checkerboard creates a narrow carrier peak and is penalized.

    Args:
        image: Predicted BCHW image at the native coordinate-query resolution.
        scale_factor: Output pixels per input pixel.
        band_width: Radius of the carrier core in cycles/output-pixel.
        ring_inner_ratio: Inner comparison-ring radius / ``band_width``.
        ring_outer_ratio: Outer comparison-ring radius / ``band_width``.
        peak_ratio: Allowed core/ring power ratio before a penalty is applied.
        harmonics: Number of coordinate-lattice harmonics to inspect.
        include_axes: Inspect horizontal and vertical carrier peaks.
        include_diagonals: Inspect diagonal checkerboard carrier peaks.
        min_frequency: Ignore carriers too close to DC.

    Returns:
        A dimensionless scalar normalized by detached total spectral power.
    """
    if image.ndim != 4:
        raise ValueError(f"Expected BCHW image, got shape {tuple(image.shape)}")
    if scale_factor <= 0:
        raise ValueError(f"scale_factor must be positive, got {scale_factor}")
    if band_width <= 0:
        raise ValueError(f"band_width must be positive, got {band_width}")
    if not 1.0 <= ring_inner_ratio < ring_outer_ratio:
        raise ValueError(
            "Expected 1 <= ring_inner_ratio < ring_outer_ratio, got "
            f"{ring_inner_ratio}, {ring_outer_ratio}"
        )
    if harmonics < 1:
        raise ValueError(f"harmonics must be >= 1, got {harmonics}")

    # FFT is kept in float32 because CUDA FFT support for half precision is
    # shape-restricted and its power estimate is less stable.
    image = image.float()
    _, _, height, width = image.shape
    window_y = torch.hann_window(
        height, periodic=False, device=image.device, dtype=image.dtype
    )
    window_x = torch.hann_window(
        width, periodic=False, device=image.device, dtype=image.dtype
    )
    window = window_y[:, None] * window_x[None, :]
    centered = image - image.mean(dim=(-2, -1), keepdim=True)
    spectrum = torch.fft.fftshift(
        torch.fft.fft2(centered * window, norm="ortho"), dim=(-2, -1)
    )
    power = spectrum.abs().square().mean(dim=1)

    fy = torch.fft.fftshift(
        torch.fft.fftfreq(height, device=image.device, dtype=image.dtype)
    )
    fx = torch.fft.fftshift(
        torch.fft.fftfreq(width, device=image.device, dtype=image.dtype)
    )
    grid_y, grid_x = torch.meshgrid(fy, fx, indexing="ij")

    frequencies: list[float] = []
    for harmonic in range(1, harmonics + 1):
        frequency = _alias_frequency(harmonic / float(scale_factor))
        if frequency < min_frequency:
            continue
        if not any(math.isclose(frequency, old, abs_tol=1e-6)
                   for old in frequencies):
            frequencies.append(frequency)

    penalties = []
    for frequency in frequencies:
        for center_y, center_x in _carrier_centers(
            frequency, include_axes, include_diagonals
        ):
            distance_y = _periodic_distance(grid_y, center_y)
            distance_x = _periodic_distance(grid_x, center_x)
            distance = torch.sqrt(distance_y.square() + distance_x.square())
            core = distance <= band_width
            ring = (
                (distance >= band_width * ring_inner_ratio)
                & (distance <= band_width * ring_outer_ratio)
            )
            if not bool(core.any()) or not bool(ring.any()):
                continue
            core_power = power[:, core].mean(dim=-1)
            ring_power = power[:, ring].mean(dim=-1)
            penalties.append(F.relu(core_power - peak_ratio * ring_power))

    if not penalties:
        return image.new_zeros(())

    carrier_excess = torch.stack(penalties, dim=0).mean(dim=0)
    total_power = power.mean(dim=(-2, -1)).detach().clamp_min(eps)
    return (carrier_excess / total_power).mean()


__all__ = ["coordinate_frequency_loss"]
