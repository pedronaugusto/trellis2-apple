from typing import *
from tqdm import tqdm
import numpy as np
import torch
import cv2
from PIL import Image
import trimesh
import trimesh.visual

import platform

_HAS_DR = False
_HAS_MESH = False
_BACKEND = None
dr = None

# Differentiable rasterizer — mtldiffrast (Metal) or nvdiffrast (CUDA)
try:
    import mtldiffrast.torch as dr
    _HAS_DR = True
    _BACKEND = 'metal'
except ImportError:
    try:
        import nvdiffrast.torch as dr
        _HAS_DR = True
        _BACKEND = 'cuda'
    except ImportError:
        pass

# Mesh processing — cumesh auto-selects Metal/CUDA
try:
    import cumesh
    _MeshBackend = cumesh.CuMesh
    _BVH = cumesh.cuBVH
    _remesh_narrow_band_dc = cumesh.remeshing.remesh_narrow_band_dc
    _HAS_MESH = True
    if _BACKEND is None:
        _BACKEND = 'metal' if platform.system() == 'Darwin' else 'cuda'
except ImportError:
    pass

_HAS_GPU_DEPS = _HAS_DR and _HAS_MESH

try:
    from flex_gemm.ops.grid_sample import grid_sample_3d as _flex_grid_sample_3d
    _HAS_FLEX_GEMM = True
except ImportError:
    _HAS_FLEX_GEMM = False


def _grid_sample_3d(feats, coords, shape, grid, mode='trilinear'):
    """Grid sampling with flex_gemm on CUDA, F.grid_sample fallback otherwise."""
    if _HAS_FLEX_GEMM:
        return _flex_grid_sample_3d(feats, coords, shape, grid, mode=mode)
    import torch.nn.functional as F_gs
    B, C = shape[0], shape[1]
    D, H, W = shape[2], shape[3], shape[4]
    device = feats.device
    dense_vol = torch.zeros(B, C, D, H, W, dtype=feats.dtype, device=device)
    batch_idx = coords[:, 0].long()
    cx, cy, cz = coords[:, 1].long(), coords[:, 2].long(), coords[:, 3].long()
    dense_vol[batch_idx, :, cx, cy, cz] = feats
    grid_norm = torch.stack([
        grid[..., 2] / (W - 1) * 2 - 1,
        grid[..., 1] / (H - 1) * 2 - 1,
        grid[..., 0] / (D - 1) * 2 - 1,
    ], dim=-1).reshape(B, 1, 1, -1, 3)
    sampled = F_gs.grid_sample(dense_vol, grid_norm, mode='bilinear',
                               align_corners=True, padding_mode='border')
    M = grid.shape[1]
    return sampled.reshape(B * C, M)


def to_glb(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    attr_volume: torch.Tensor,
    coords: torch.Tensor,
    attr_layout: Dict[str, slice],
    aabb: Union[list, tuple, np.ndarray, torch.Tensor],
    voxel_size: Union[float, list, tuple, np.ndarray, torch.Tensor] = None,
    grid_size: Union[int, list, tuple, np.ndarray, torch.Tensor] = None,
    decimation_target: int = 1000000,
    texture_size: int = 2048,
    remesh: bool = False,
    remesh_band: float = 1,
    remesh_project: float = 0.9,
    mesh_cluster_threshold_cone_half_angle_rad=np.radians(90.0),
    mesh_cluster_refine_iterations=0,
    mesh_cluster_global_iterations=1,
    mesh_cluster_smooth_strength=1,
    # Default OPAQUE, not 'auto'. A glTF BLEND material does not write the
    # depth buffer, so the *entire* surface stops occluding — including the
    # fully-opaque body around a few genuinely transparent texels. The result
    # reads as flipped/random normals (you see into the interior) even though
    # the geometry is sound. Verified on the kei asset: winding disagreement
    # 0.019% and 100% of from-outside rays hit a front-facing triangle, yet it
    # rendered see-through purely because ~4% low-alpha glass texels tripped
    # BLEND. Upstream TRELLIS.2 also keeps alpha in the texture but inactive.
    # Pass alpha_mode='auto' to opt back into detection, or 'BLEND' to force it.
    alpha_mode: str = 'OPAQUE',
    alpha_blend_threshold: float = 0.5,
    alpha_blend_min_fraction: float = 0.01,
    verbose: bool = False,
    use_tqdm: bool = False,
):
    """
    Convert an extracted mesh to a GLB file.
    Performs cleaning, optional remeshing, UV unwrapping, and texture baking from a volume.

    Args:
        vertices: (N, 3) tensor of vertex positions
        faces: (M, 3) tensor of vertex indices
        attr_volume: (L, C) features of a sprase tensor for attribute interpolation
        coords: (L, 3) tensor of coordinates for each voxel
        attr_layout: dictionary of slice objects for each attribute
        aabb: (2, 3) tensor of minimum and maximum coordinates of the volume
        voxel_size: (3,) tensor of size of each voxel
        grid_size: (3,) tensor of number of voxels in each dimension
        decimation_target: target number of vertices for mesh simplification
        texture_size: size of the texture for baking
        remesh: whether to perform remeshing
        remesh_band: size of the remeshing band
        remesh_project: projection factor for remeshing
        mesh_cluster_threshold_cone_half_angle_rad: threshold for cone-based clustering in uv unwrapping
        mesh_cluster_refine_iterations: number of iterations for refining clusters in uv unwrapping
        mesh_cluster_global_iterations: number of global iterations for clustering in uv unwrapping
        mesh_cluster_smooth_strength: strength of smoothing for clustering in uv unwrapping
        alpha_mode: 'auto' (default), 'OPAQUE', 'BLEND', or 'MASK'. 'auto' selects
            BLEND only if more than `alpha_blend_min_fraction` of valid texels have
            alpha below `alpha_blend_threshold`; otherwise OPAQUE. Pass 'OPAQUE' or
            'BLEND' explicitly to override.
        alpha_blend_threshold: in [0, 1]. With `alpha_mode='auto'`, a texel counts as
            transparent if its baked alpha is below this fraction. Defaults to 0.5
            so accidental sub-1.0 alpha values from numerical drift in the flow
            model don't accidentally trigger BLEND mode.
        alpha_blend_min_fraction: in [0, 1]. With `alpha_mode='auto'`, BLEND is
            chosen only if at least this fraction of valid texels are transparent.
            Defaults to 0.01 (1%) so a handful of outlier texels does not flip
            the whole material to transparent.
        verbose: whether to print verbose messages
        use_tqdm: whether to use tqdm to display progress bar
    """
    # Auto-fallback to CPU pipeline when no GPU deps available
    if not _HAS_GPU_DEPS:
        from .postprocess_cpu import to_glb as to_glb_cpu
        return to_glb_cpu(
            vertices=vertices, faces=faces, attr_volume=attr_volume,
            coords=coords, attr_layout=attr_layout, aabb=aabb,
            voxel_size=voxel_size, grid_size=grid_size,
            decimation_target=decimation_target, texture_size=texture_size,
            remesh=remesh, remesh_band=remesh_band, remesh_project=remesh_project,
            alpha_mode=alpha_mode, alpha_blend_threshold=alpha_blend_threshold,
            alpha_blend_min_fraction=alpha_blend_min_fraction,
            verbose=verbose, use_tqdm=use_tqdm,
        )

    # Select device based on backend
    # Metal path: all GPU compute goes through Metal kernels directly (mtldiffrast,
    # mtlbvh, cumesh, flex_gemm). CPU tensors on Apple Silicon unified memory are
    # directly GPU-accessible — no MPS overhead needed.
    if _BACKEND == 'metal':
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')

    # --- Input Normalization (AABB, Voxel Size, Grid Size) ---
    if isinstance(aabb, (list, tuple)):
        aabb = np.array(aabb)
    if isinstance(aabb, np.ndarray):
        aabb = torch.tensor(aabb, dtype=torch.float32, device=coords.device)
    assert isinstance(aabb, torch.Tensor), f"aabb must be a list, tuple, np.ndarray, or torch.Tensor, but got {type(aabb)}"
    assert aabb.dim() == 2, f"aabb must be a 2D tensor, but got {aabb.shape}"
    assert aabb.size(0) == 2, f"aabb must have 2 rows, but got {aabb.size(0)}"
    assert aabb.size(1) == 3, f"aabb must have 3 columns, but got {aabb.size(1)}"

    # Calculate grid dimensions based on AABB and voxel size
    if voxel_size is not None:
        if isinstance(voxel_size, float):
            voxel_size = [voxel_size, voxel_size, voxel_size]
        if isinstance(voxel_size, (list, tuple)):
            voxel_size = np.array(voxel_size)
        if isinstance(voxel_size, np.ndarray):
            voxel_size = torch.tensor(voxel_size, dtype=torch.float32, device=coords.device)
        grid_size = ((aabb[1] - aabb[0]) / voxel_size).round().int()
    else:
        assert grid_size is not None, "Either voxel_size or grid_size must be provided"
        if isinstance(grid_size, int):
            grid_size = [grid_size, grid_size, grid_size]
        if isinstance(grid_size, (list, tuple)):
            grid_size = np.array(grid_size)
        if isinstance(grid_size, np.ndarray):
            grid_size = torch.tensor(grid_size, dtype=torch.int32, device=coords.device)
        voxel_size = (aabb[1] - aabb[0]) / grid_size
    
    # Assertions for dimensions
    assert isinstance(voxel_size, torch.Tensor)
    assert voxel_size.dim() == 1 and voxel_size.size(0) == 3
    assert isinstance(grid_size, torch.Tensor)
    assert grid_size.dim() == 1 and grid_size.size(0) == 3
    
    if use_tqdm:
        pbar = tqdm(total=6, desc="Extracting GLB")
    if verbose:
        print(f"Original mesh: {vertices.shape[0]} vertices, {faces.shape[0]} faces")

    # Move data to the backend device. All of these tensors meet in downstream
    # ops (remesh center/scale from aabb, attribute grid-sampling), so every
    # one must land on `device` — pipeline outputs arrive on MPS while the
    # Metal backend computes on cpu tensors (unified memory).
    vertices = vertices.to(device)
    faces = faces.to(device)
    coords = coords.to(device)
    attr_volume = attr_volume.to(device)
    aabb = aabb.to(device)
    voxel_size = voxel_size.to(device)
    grid_size = grid_size.to(device)

    # Initialize mesh handler
    mesh = _MeshBackend()
    mesh.init(vertices, faces)
    
    # --- Initial Mesh Cleaning ---
    # Fills holes as much as we can before processing
    mesh.fill_holes(max_hole_perimeter=3e-2)
    if verbose:
        print(f"After filling holes: {mesh.num_vertices} vertices, {mesh.num_faces} faces")
    vertices, faces = mesh.read()
    if use_tqdm:
        pbar.update(1)
        
    # Build BVH for the current mesh to guide remeshing
    if use_tqdm:
        pbar.set_description("Building BVH")
    if verbose:
        print(f"Building BVH for current mesh...", end='', flush=True)
    bvh = _BVH(vertices, faces)
    if use_tqdm:
        pbar.update(1)
    if verbose:
        print("Done")
        
    if use_tqdm:
        pbar.set_description("Cleaning mesh")
    if verbose:
        print("Cleaning mesh...")
    
    # --- Branch 1: Standard Pipeline (Simplification & Cleaning) ---
    if not remesh:
        # Step 1: Aggressive simplification (3x target)
        mesh.simplify(decimation_target * 3, verbose=verbose)
        if verbose:
            print(f"After inital simplification: {mesh.num_vertices} vertices, {mesh.num_faces} faces")
        
        # Step 2: Clean up topology (duplicates, non-manifolds, isolated parts)
        mesh.remove_duplicate_faces()
        mesh.repair_non_manifold_edges()
        mesh.remove_small_connected_components(1e-5)
        mesh.fill_holes(max_hole_perimeter=3e-2)
        if verbose:
            print(f"After initial cleanup: {mesh.num_vertices} vertices, {mesh.num_faces} faces")
            
        # Step 3: Final simplification to target count
        mesh.simplify(decimation_target, verbose=verbose)
        if verbose:
            print(f"After final simplification: {mesh.num_vertices} vertices, {mesh.num_faces} faces")
        
        # Step 4: Final Cleanup loop
        mesh.remove_duplicate_faces()
        mesh.repair_non_manifold_edges()
        mesh.remove_small_connected_components(1e-5)
        mesh.fill_holes(max_hole_perimeter=3e-2)
        if verbose:
            print(f"After final cleanup: {mesh.num_vertices} vertices, {mesh.num_faces} faces")
            
        # Step 5: Unify face orientations
        mesh.unify_face_orientations()
    
    # --- Branch 2: Remeshing Pipeline ---
    else:
        center = aabb.mean(dim=0)
        scale = (aabb[1] - aabb[0]).max().item()
        resolution = grid_size.max().item()
        
        # Perform Dual Contouring remeshing (rebuilds topology)
        mesh.init(*_remesh_narrow_band_dc(
            vertices, faces,
            center = center,
            scale = (resolution + 3 * remesh_band) / resolution * scale,
            resolution = resolution,
            band = remesh_band,
            project_back = remesh_project, # Snaps vertices back to original surface
            verbose = verbose,
            bvh = bvh,
        ))
        if verbose:
            print(f"After remeshing: {mesh.num_vertices} vertices, {mesh.num_faces} faces")

        # Clean up topology before simplification
        mesh.remove_duplicate_faces()
        mesh.repair_non_manifold_edges()
        mesh.remove_small_connected_components(1e-5)
        mesh.fill_holes(max_hole_perimeter=3e-2)
        if verbose:
            print(f"After cleanup: {mesh.num_vertices} vertices, {mesh.num_faces} faces")

        # Simplify and clean the remeshed result
        mesh.simplify(decimation_target, verbose=verbose)
        if verbose:
            print(f"After simplifying: {mesh.num_vertices} vertices, {mesh.num_faces} faces")
    
    if use_tqdm:
        pbar.update(1)
    if verbose:
        print("Done")
        
    
    # --- UV Parameterization ---
    if use_tqdm:
        pbar.set_description("Parameterizing new mesh")
    if verbose:
        print("Parameterizing new mesh...")
    
    out_vertices, out_faces, out_uvs, out_vmaps = mesh.uv_unwrap(
        compute_charts_kwargs={
            "threshold_cone_half_angle_rad": mesh_cluster_threshold_cone_half_angle_rad,
            "refine_iterations": mesh_cluster_refine_iterations,
            "global_iterations": mesh_cluster_global_iterations,
            "smooth_strength": mesh_cluster_smooth_strength,
        },
        return_vmaps=True,
        verbose=verbose,
    )
    out_vertices = out_vertices.to(device)
    out_faces = out_faces.to(device)
    out_uvs = out_uvs.to(device)
    out_vmaps = out_vmaps.to(device)
    mesh.compute_vertex_normals()
    out_normals = mesh.read_vertex_normals()[out_vmaps]
    
    if use_tqdm:
        pbar.update(1)
    if verbose:
        print("Done")
    
    # --- Texture Baking (Attribute Sampling) ---
    if use_tqdm:
        pbar.set_description("Sampling attributes")
    if verbose:
        print("Sampling attributes...", end='', flush=True)
        
    # Setup differentiable rasterizer context
    ctx = dr.MtlRasterizeContext() if _BACKEND == 'metal' else dr.RasterizeCudaContext()
    # Prepare UV coordinates for rasterization (rendering in UV space)
    uvs_rast = torch.cat([out_uvs * 2 - 1, torch.zeros_like(out_uvs[:, :1]), torch.ones_like(out_uvs[:, :1])], dim=-1).unsqueeze(0)
    rast = torch.zeros((1, texture_size, texture_size, 4), device=device, dtype=torch.float32)
    
    # Rasterize in chunks to save memory
    for i in range(0, out_faces.shape[0], 100000):
        rast_chunk, _ = dr.rasterize(
            ctx, uvs_rast, out_faces[i:i+100000],
            resolution=[texture_size, texture_size],
        )
        mask_chunk = rast_chunk[..., 3:4] > 0
        rast_chunk[..., 3:4] += i # Store face ID in alpha channel
        rast = torch.where(mask_chunk, rast_chunk, rast)
    
    # Mask of valid pixels in texture
    mask = rast[0, ..., 3] > 0
    
    # Interpolate 3D positions in UV space (finding 3D coord for every texel)
    pos = dr.interpolate(out_vertices.unsqueeze(0), rast, out_faces)[0][0]
    valid_pos = pos[mask]
    
    # Map these positions back to the *original* high-res mesh to get accurate attributes
    # This corrects geometric errors introduced by simplification/remeshing
    _, face_id, uvw = bvh.unsigned_distance(valid_pos, return_uvw=True)
    orig_tri_verts = vertices[faces[face_id.long()]] # (N_new, 3, 3)
    valid_pos = (orig_tri_verts * uvw.unsqueeze(-1)).sum(dim=1)
    
    # Trilinear sampling from the attribute volume (Color, Material props)
    attrs = torch.zeros(texture_size, texture_size, attr_volume.shape[1], device=device)
    attrs[mask] = _grid_sample_3d(
        attr_volume,
        torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=-1),
        shape=torch.Size([1, attr_volume.shape[1], *grid_size.tolist()]),
        grid=((valid_pos - aabb[0]) / voxel_size).reshape(1, -1, 3),
        mode='trilinear',
    )
    if use_tqdm:
        pbar.update(1)
    if verbose:
        print("Done")
    
    # --- Texture Post-Processing & Material Construction ---
    if use_tqdm:
        pbar.set_description("Finalizing mesh")
    if verbose:
        print("Finalizing mesh...", end='', flush=True)
    
    mask = mask.cpu().numpy()
    
    # Extract channels based on layout (BaseColor, Metallic, Roughness, Alpha)
    base_color = np.clip(attrs[..., attr_layout['base_color']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
    metallic = np.clip(attrs[..., attr_layout['metallic']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
    roughness = np.clip(attrs[..., attr_layout['roughness']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
    alpha = np.clip(attrs[..., attr_layout['alpha']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
    # Resolve alphaMode. The previous behavior was a hair-trigger min-based
    # detection (any single texel with alpha < 250 → BLEND), which is fragile
    # under numerical drift in the alpha attribute (e.g. bf16 flow models on
    # Apple Silicon can leave a handful of texels with sub-1.0 alpha that are
    # not semantically transparent). The 'auto' default now requires a
    # meaningful *fraction* of texels to be transparent before flipping.
    valid_modes = ('auto', 'OPAQUE', 'BLEND', 'MASK')
    if alpha_mode not in valid_modes:
        raise ValueError(
            f"alpha_mode must be one of {valid_modes}, got {alpha_mode!r}"
        )
    if alpha_mode == 'auto':
        alpha_valid = alpha[mask]
        threshold_u8 = int(round(np.clip(alpha_blend_threshold, 0.0, 1.0) * 255))
        if alpha_valid.size > 0:
            transparent_frac = float((alpha_valid < threshold_u8).mean())
        else:
            transparent_frac = 0.0
        if transparent_frac > alpha_blend_min_fraction:
            resolved_alpha_mode = 'BLEND'
            if verbose:
                print(
                    f"Detected transparency: {transparent_frac:.2%} of texels "
                    f"have alpha < {alpha_blend_threshold:.2f} "
                    f"(threshold {alpha_blend_min_fraction:.2%}); using BLEND mode"
                )
        else:
            resolved_alpha_mode = 'OPAQUE'
            if verbose and alpha_valid.size > 0:
                print(
                    f"No significant transparency: {transparent_frac:.2%} of "
                    f"texels below threshold {alpha_blend_threshold:.2f}; "
                    f"using OPAQUE mode"
                )
    else:
        resolved_alpha_mode = alpha_mode
    
    # Inpainting: fill gaps (dilation) to prevent black seams at UV boundaries
    mask_inv = (~mask).astype(np.uint8)
    base_color = cv2.inpaint(base_color, mask_inv, 3, cv2.INPAINT_TELEA)
    metallic = cv2.inpaint(metallic, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]
    roughness = cv2.inpaint(roughness, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]
    alpha = cv2.inpaint(alpha, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]
    
    # Create PBR material
    # Standard PBR packs Metallic and Roughness into Blue and Green channels
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.fromarray(np.concatenate([base_color, alpha], axis=-1)),
        baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
        metallicRoughnessTexture=Image.fromarray(np.concatenate([np.zeros_like(metallic), roughness, metallic], axis=-1)),
        metallicFactor=1.0,
        roughnessFactor=1.0,
        alphaMode=resolved_alpha_mode,
        doubleSided=True,
    )
    
    # --- Coordinate System Conversion & Final Object ---
    vertices_np = out_vertices.cpu().numpy()
    faces_np = out_faces.cpu().numpy()
    uvs_np = out_uvs.cpu().numpy()
    normals_np = out_normals.cpu().numpy()
    
    # Y-up to Z-up for GLB (must copy to avoid in-place corruption)
    vertices_np[:, 1], vertices_np[:, 2] = vertices_np[:, 2].copy(), -vertices_np[:, 1].copy()
    normals_np[:, 1], normals_np[:, 2] = normals_np[:, 2].copy(), -normals_np[:, 1].copy()
    uvs_np[:, 1] = 1 - uvs_np[:, 1]
    
    textured_mesh = trimesh.Trimesh(
        vertices=vertices_np,
        faces=faces_np,
        vertex_normals=normals_np,
        process=False,
        visual=trimesh.visual.TextureVisuals(uv=uvs_np, material=material)
    )
    
    if use_tqdm:
        pbar.update(1)
        pbar.close()
    if verbose:
        print("Done")
    
    return textured_mesh