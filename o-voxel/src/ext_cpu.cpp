/*
 * CPU-only extension entry point (no CUDA): convert + SVO io bindings.
 * Mirrors ext.cpp minus hash/serialize/rasterize, which are GPU-backed.
 */
#include <torch/extension.h>
#include "convert/api.h"
#include "io/api.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Convert functions
    m.def("mesh_to_flexible_dual_grid_cpu", &mesh_to_flexible_dual_grid_cpu, py::call_guard<py::gil_scoped_release>());
    m.def("textured_mesh_to_volumetric_attr_cpu", &textured_mesh_to_volumetric_attr_cpu, py::call_guard<py::gil_scoped_release>());
    // IO functions
    m.def("encode_sparse_voxel_octree_cpu", &encode_sparse_voxel_octree_cpu, py::call_guard<py::gil_scoped_release>());
    m.def("decode_sparse_voxel_octree_cpu", &decode_sparse_voxel_octree_cpu, py::call_guard<py::gil_scoped_release>());
    m.def("encode_sparse_voxel_octree_attr_parent_cpu", &encode_sparse_voxel_octree_attr_parent_cpu, py::call_guard<py::gil_scoped_release>());
    m.def("decode_sparse_voxel_octree_attr_parent_cpu", &decode_sparse_voxel_octree_attr_parent_cpu, py::call_guard<py::gil_scoped_release>());
    m.def("encode_sparse_voxel_octree_attr_neighbor_cpu", &encode_sparse_voxel_octree_attr_neighbor_cpu, py::call_guard<py::gil_scoped_release>());
    m.def("decode_sparse_voxel_octree_attr_neighbor_cpu", &decode_sparse_voxel_octree_attr_neighbor_cpu, py::call_guard<py::gil_scoped_release>());
}
