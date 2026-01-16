# SPDX-License-Identifier: MIT
#
# Copyright (c) 2025 Ioannis Farmakis
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import open3d as o3d
from functools import partial
from operator import is_not


def az2vec(az):
    """ 
    ::Converts azimuth in degrees to 2D normalized vector
    """
    return [np.sin(np.deg2rad(az)), np.cos(np.deg2rad(az))]


def cluster_connected_triangles(mesh):
    """ 
    ::Returns larger connected component of mesh triangles
    """
    ids, size, _ = mesh.cluster_connected_triangles()
    max_id = np.argmax(size)
    triangles = np.asarray(mesh.triangles)[np.asarray(ids)==max_id]
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    return mesh


def unit_vector(vector):
    """ 
    ::Returns the unit vector of the vector
    """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """
    ::Returns the angle in radians between vectors 'v1' and 'v2'
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def filter_empty_space(voxel_keys, min_bound, max_bound):
    """
    ::Function to filter out empty clusters that are not between
      the two surfaces
    Input:
    voxel_keys -> empty voxel indices (nx3 array)
    min_bound -> minimum window point (1x3 array)
    max_bound -> maximum window point (1x3 array)
    Output: valid voxel indices (nx3 array)
    """
    keys_pcd = o3d.geometry.PointCloud()
    keys_pcd.points = o3d.utility.Vector3dVector(np.asarray(voxel_keys))
    labels = np.array(keys_pcd.cluster_dbscan(eps=1.1, min_points=2))
    empty_keys = np.asarray(keys_pcd.points)
    minp = np.where((empty_keys==min_bound).all(axis=1))
    maxp = np.where((empty_keys==max_bound).all(axis=1))
    idA = labels[minp]
    idB = labels[maxp]
    labels[np.where(labels==idB)] = idA
    empty_keys = np.asarray(keys_pcd.points)[labels > idA]
    return empty_keys


def find_adjacents(voxel_grid, facets_only=False):
    """
    ::Function to find the neighboring voxels to voxel grid
    Input:
    voxel_grid -> input voxel grid indices (nx3 array)
    facets_only -> whether all or only facet neighbors are searched (bool) 
    Output: indices of neighboring voxels (nx3 array)
    """
    if not facets_only:
        # 26-connectivity
        adjacency_matrix = [
            [1, 0, 0], [-1, 0, 0], [0, 1, 0],
            [0, -1, 0], [0, 0, 1], [0, 0, -1],
            [1, 1, 0], [-1, 1, 0], [1, -1, 0],
            [-1, -1, 0], [0, 1 ,1], [0, 1 ,-1],
            [0, -1 ,1], [0, -1 ,-1], [1, 0, 1],
            [1, 0, -1], [-1, 0, 1], [-1, 0, -1],
            [1, 1, 1], [-1, -1, -1], [1, 1, -1],
            [1, -1, 1], [-1, 1, 1], [1, -1, -1],
            [-1, -1, 1], [-1, 1, -1]
        ]
    else:
        # 6-connectivity
        adjacency_matrix = [
            [1, 0, 0], [-1, 0, 0], [0, 1, 0],
            [0, -1, 0], [0, 0, 1], [0, 0, -1],
        ]

    adjacents = [voxel_grid + neighbor for neighbor in adjacency_matrix]
    adjacents = np.asarray(adjacents).reshape(-1, 3)
    adjacents_pcd = o3d.geometry.PointCloud()
    adjacents_pcd.points = o3d.utility.Vector3dVector(np.asarray(adjacents))
    adjacents_pcd.remove_duplicated_points()
    adjacents = np.asarray(adjacents_pcd.points)
    adjacents, counts = np.unique(np.vstack((adjacents, voxel_grid)), axis=0, return_counts=True)
    adjacents = adjacents[np.where(counts==1)[0]]
    return adjacents


def compute_volume(inner, outter, voxel_size):
    """
    ::Function to compute the volume of a voxel-based empty space cluster
    Input:
    inner -> inner voxels indices (nx3 array)
    outter -> neighboring voxels indices (nx3 array)
    voxel_size -> the size of a voxel (float)
    Output: 
    volume -> volume of the cluster (float)
    bias -> approximate volume uncertainty (float)
    """
    error = (np.power(voxel_size,3) * len(outter)) / 2
    volume = (np.power(voxel_size,3) * len(inner)) + error
    bias = error/2
    return volume, bias


def get_reverse_rotation_matrix(R):
    """
    ::Function to reverse a 3x3 rotation matrix
    Input: 3x3 array
    Output: 3x3 array
    """
    reverse_mask = np.array((
                    [1,-1,-1], 
                    [-1,1,-1], 
                    [-1,-1,1])
                    )
    return R * reverse_mask


def create_3d_object(voxels, scale, min_bound):
    """
    ::Function to create 3d object of a voxel grid
    Input:
    voxels -> voxel indices (nx3 array)
    scale -> voxel size (float)
    min_bound -> min bound of the entire scene on current view (1x3 array)
    Output: open3d mesh object
    """
    cluster = o3d.geometry.PointCloud()
    cluster.points = o3d.utility.Vector3dVector(voxels)
    cluster = cluster.translate(min_bound)  # translate grid to the current view position
    cluster = cluster.scale(scale, min_bound)  # scale grid to actual size

    # create mesh from voxel grid
    out_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(cluster, scale)
    mesh = o3d.geometry.TriangleMesh()
    for voxel in out_grid.get_voxels():
        box = o3d.geometry.TriangleMesh.create_box()
        box = box.translate(voxel.grid_index, relative=False)
        mesh += box  
    mesh = mesh.scale(scale, mesh.get_center())
    mesh = mesh.translate(cluster.get_center(), relative=False)
    mesh.compute_vertex_normals()

    return mesh


def check_change_type(surface, block):
    """
    ::Function to check if a block corresponds to loss volume
    Input:
    surface -> before triangle mesh (open3d object)
    block -> block triangle mesh (open3d object
    Output: str ('loss' or 'gain')
    """
    bb = block.get_axis_aligned_bounding_box()
    max_y = bb.get_max_bound()[1]
    min_y = bb.get_min_bound()[1]
    bb.scale(0.2, bb.get_center())
    max_bound = bb.get_max_bound()
    max_bound[1] = max_y + (max_y-min_y)/2
    bb.max_bound = max_bound
    crop = surface.crop(bb)

    if crop.is_empty():
        return 'gain'
    else:
        return 'loss'


def get_shape(mesh):
    """
    ::Function to get the 3-axis shape of a block
    """
    mesh.compute_triangle_normals()
    bb = mesh.get_oriented_bounding_box()
    return np.sort(bb.extent)[::-1]


def visualize_voxel_grid(voxel_keys, tag, color=[0,0,1], scale=None, translate=np.array([None]), save=True):
    """
    ::Function to visualize a grid incices array as 3D mesh
    """
    grid_pcd = o3d.geometry.PointCloud()
    grid_pcd.points = o3d.utility.Vector3dVector(np.asarray(voxel_keys))
    grid_pcd.paint_uniform_color(color)

    if scale: 
        grid_pcd = grid_pcd.scale(scale, np.array([0,0,0]))
    if translate.any(): 
        grid_pcd = grid_pcd.translate(translate)

    if save: o3d.io.write_point_cloud("{}_grid_pcd.pcd".format(tag), grid_pcd)

    grid = o3d.geometry.VoxelGrid.create_from_point_cloud(grid_pcd, scale if scale else 1)
    mesh = o3d.geometry.TriangleMesh()
    for voxel in grid.get_voxels():
        box = o3d.geometry.TriangleMesh.create_box()
        box = box.translate(voxel.grid_index, relative=False)
        mesh += box
    mesh.compute_vertex_normals()

    if scale: 
        mesh = mesh.scale(scale, np.array([0,0,0]))
    if translate.any(): 
        mesh = mesh.translate(grid_pcd.get_center(), relative=False)

    if save: o3d.io.write_triangle_mesh("{}_grid_mesh.ply".format(tag), mesh)
    if not save: o3d.visualization.draw_geometries([mesh])










