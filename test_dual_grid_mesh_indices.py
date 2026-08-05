"""Dual-grid mesh extraction must never emit a sentinel index into the faces.

Regression test. `flexible_dual_grid_to_mesh` looks up the four voxels around
each edge and keeps only the quads whose four corners all exist; a corner that
does not exist comes back as 0xffffffff. The mask that dropped those quads was
`connected_voxel_indices != 0xffffffff`, applied to an int32 tensor.

CPU and CUDA wrap an out-of-range Python scalar into the tensor's dtype, so
0xffffffff becomes -1 and the comparison matches. MPS compares in a wider type,
so -1 != 4294967295 held and the mask was unconditionally all-True: every quad
with a missing corner survived, carrying -1 into the face array. Because a
negative index wraps, all of them fanned out to the last vertex, producing long
radial strands across the whole model. In a 512-grid generation that was ~2.3k
faces whose median edge was 200x longer than a normal one.

The fix tests the sign instead, which is dtype- and device-independent. These
tests assert the property that was violated: no out-of-range index reaches the
faces, and MPS extraction is identical to CPU extraction.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "o-voxel"))

import pytest
import torch

from o_voxel.convert.flexible_dual_grid import flexible_dual_grid_to_mesh

GRID_SIZE = 8
AABB = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]

DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])

CASES = {
    # Every edge-neighbour lookup misses, so no quad is complete and the mesh
    # must come out empty. This is the case that produced the strands.
    "isolated_voxel": [[4, 4, 4]],
    # Solid block: the interior quads are fully populated and must survive,
    # which is what keeps the fix from simply rejecting everything.
    "solid_2x2x2": [[x, y, z] for x in (4, 5) for y in (4, 5) for z in (4, 5)],
    # Two disjoint blocks: complete quads and missing corners in one call.
    "two_disjoint_blocks": (
        [[x, y, z] for x in (1, 2) for y in (1, 2) for z in (1, 2)]
        + [[x, y, z] for x in (5, 6) for y in (5, 6) for z in (5, 6)]
    ),
}


def _extract(coords_list, device):
    """Run extraction on `device` with all voxels flagged intersected."""
    # The static offset/split tables are memoized on the function and bound to
    # the device they were first built on, so they must be dropped when the
    # test moves between devices.
    for attr in (
        "edge_neighbor_voxel_offset",
        "quad_split_1",
        "quad_split_2",
        "quad_split_train",
    ):
        if hasattr(flexible_dual_grid_to_mesh, attr):
            delattr(flexible_dual_grid_to_mesh, attr)

    n = len(coords_list)
    coords = torch.tensor(coords_list, dtype=torch.int32, device=device)
    dual_vertices = torch.full((n, 3), 0.5, device=device)
    intersected = torch.ones((n, 3), dtype=torch.bool, device=device)
    split_weight = torch.ones((n, 1), device=device)

    vertices, faces = flexible_dual_grid_to_mesh(
        coords,
        dual_vertices,
        intersected,
        split_weight,
        aabb=AABB,
        grid_size=GRID_SIZE,
    )
    return vertices.cpu(), faces.cpu()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("case", list(CASES))
def test_faces_index_only_real_vertices(case, device):
    vertices, faces = _extract(CASES[case], device)
    assert faces.numel() == 0 or faces.min() >= 0, (
        f"{device}/{case}: {int((faces < 0).sum())} negative face indices; a "
        f"missing-voxel sentinel reached the face array"
    )
    assert faces.numel() == 0 or faces.max() < vertices.shape[0], (
        f"{device}/{case}: face index >= {vertices.shape[0]} vertices"
    )


@pytest.mark.parametrize("device", DEVICES)
def test_isolated_voxel_yields_no_faces(device):
    """A voxel with no complete quad contributes nothing, on every device."""
    _, faces = _extract(CASES["isolated_voxel"], device)
    assert faces.shape[0] == 0, f"{device}: expected no faces, got {faces.shape[0]}"


@pytest.mark.skipif("mps" not in DEVICES, reason="MPS not available")
@pytest.mark.parametrize("case", list(CASES))
def test_mps_extraction_matches_cpu(case):
    cpu_v, cpu_f = _extract(CASES[case], "cpu")
    mps_v, mps_f = _extract(CASES[case], "mps")
    assert mps_f.shape == cpu_f.shape, (
        f"{case}: MPS produced {mps_f.shape[0]} faces, CPU produced {cpu_f.shape[0]}"
    )
    assert torch.equal(mps_f, cpu_f), f"{case}: MPS and CPU face indices differ"
    assert torch.allclose(mps_v, cpu_v, atol=1e-6), f"{case}: vertex positions differ"


@pytest.mark.parametrize("device", DEVICES)
def test_narrowed_sentinel_is_negative(device):
    """The predicate the fix relies on holds on every device.

    0xffffffff narrowed to int32 is -1 regardless of whether the lookup handed
    back uint32 (CUDA) or int64 (the pure-PyTorch fallback), so a sign test
    identifies the sentinel without depending on how a backend compares an
    out-of-range Python scalar.
    """
    for dtype in (torch.int64, torch.uint32):
        sentinel = torch.tensor([0xFFFFFFFF], dtype=dtype, device=device).int()
        assert sentinel.item() == -1
        assert not bool((sentinel >= 0).item())

    real = torch.tensor([0, 1, 7], dtype=torch.int64, device=device).int()
    assert bool((real >= 0).all().item())
