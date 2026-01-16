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

import open3d as o3d
import numpy as np
import utility
import copy
import os
import xlsxwriter
from operator import is_not
from functools import partial


np.random.seed(1234)

def import_point_cloud_data(data_dir, iter, noise=None):
    """
    ::Function to merge the two input models
    Input:
    data_dir -> input/output data directory (str)
    iter -> compared pair count in input directory (int)
    Output: PCD object
    """
    in_data = os.listdir(os.path.join(data_dir, 'input'))
    before = os.path.join(data_dir, 'input', in_data[iter])
    after = os.path.join(data_dir, 'input', in_data[iter+1])
    data1 = o3d.io.read_point_cloud(before)
    data2 = o3d.io.read_point_cloud(after)
    if noise:
        data1_noise = np.random.normal(0, noise, np.asarray(data1.points).shape)
        data2_noise = np.random.normal(0, noise, np.asarray(data2.points).shape)
        data1.points = o3d.utility.Vector3dVector(np.asarray(data1.points) + data1_noise)
        data2.points = o3d.utility.Vector3dVector(np.asarray(data2.points) + data2_noise)

    return data1, data2


def import_triangle_mesh_data(data_dir, iter):
    """
    ::Function to merge the two input models
    Input:
    data_dir -> input/output data directory (str)
    iter -> compared pair count in input directory (int)
    Output: PCD object
    """
    in_data = os.listdir(os.path.join(data_dir, 'input'))
    in_data.sort()
    before = os.path.join(data_dir, 'input', in_data[iter])
    after = os.path.join(data_dir, 'input', in_data[iter+1])
    # data1 = utility.cluster_connected_triangles(o3d.io.read_triangle_mesh(before))
    # data2 = utility.cluster_connected_triangles(o3d.io.read_triangle_mesh(after))
    data1 = o3d.io.read_triangle_mesh(before)
    data2 = o3d.io.read_triangle_mesh(after)

    return data1, data2


def sample_uniform_points(mesh, voxel_size):
    """
    ::Function to uniformly sample points from the mesh surface
    making sure that each voxel would contain points
    """
    surface_area = mesh.get_surface_area()
    sample_resolution = voxel_size / 5
    sample_pts_num = int((1/sample_resolution)**2 * surface_area) * 5
    pcd = mesh.sample_points_uniformly(sample_pts_num)
    pcd = pcd.voxel_down_sample(voxel_size * 0.2)
    return pcd


def align_with_xz_view(pcd1, pcd2, az):
    """
    ::Function to orient pcd towards XZ view
    Input:
    pcd -> PCD object to be rotated
    az -> slope azimuth in degrees
    Output: 
    pcd -> rotated PCD (PCD object)
    R -> the rotation matrix (3x3 array)
    origin -> the pivot point (1x3 array)
    """
    direction = utility.az2vec(az)
    Z_rot = utility.angle_between(direction, [0,1])
    origin = pcd1.get_min_bound()
    R = pcd1.get_rotation_matrix_from_xyz((0, 0, Z_rot))
    pcd1 = pcd1.rotate(R, center=origin)
    pcd2 = pcd2.rotate(R, center=origin)
    return pcd1, pcd2, R, origin


def crop_windows(mesh, voxel_size, size):
    """
    ::Function to partion the scene into smaller windows
      for paralle processing
    Input:
    pcd -> PCD object to be partioned
    size -> step X step window size
    Output: list containing window points (list of nx3 arrays)
    """
    scene_extent = mesh.get_max_bound() - mesh.get_min_bound()
    steps = (scene_extent // size).astype(int)  # divisions along each axis
    # margin = int(size / voxel_size)

    # create the first window as a box placed at the min bound of the scene
    bb = o3d.geometry.AxisAlignedBoundingBox() 
    bb.min_bound = mesh.get_min_bound()
    bb.max_bound = mesh.get_min_bound() + np.array([size, scene_extent[1]+2, size])

    windows = []  # initiate the windows list
    for x in range(steps[0]+1):
        # the whole length of Y axis is taken in each window as is the depth of the view
        for z in range(steps[2]+1):
            # move the box (eye) to all the step position and crop points from 
            # the input scene
            eye = copy.deepcopy(bb).translate(np.array([x*size, 0, z*size]))
            scaled_eye = copy.deepcopy(eye).scale(1.5, eye.get_center())
            crop = mesh.crop(scaled_eye)
            if not crop.is_empty():
                voxel_grid = o3d.geometry.VoxelGrid()
                voxel_grid = voxel_grid.create_from_triangle_mesh_within_bounds(crop, voxel_size, mesh.get_min_bound(), mesh.get_max_bound())
                # voxel_grid = voxel_grid.create_from_point_cloud_within_bounds(crop, voxel_size, mesh.get_min_bound(), mesh.get_max_bound())
                window = np.asarray([voxel.grid_index for voxel in voxel_grid.get_voxels()])
                # window = window + ((eye.get_min_bound() - mesh.get_min_bound()) // voxel_size).astype(int)
                # window = window + (margin * x, 0, margin * z)

                # if the eye contain points, put them in the output list
                if len(window) > 0 :
                    windows.append(window)
    return windows


def fit_occupancy_grid(non_empty_voxel_keys, return_grid):
    """
    ::Function to fit a 3D occupancy grid on the input points
      within each cropped window
    Input:
    non_empty_voxel_keys -> input points (nx3 array)
    voxel_size -> resolution of the voxel grid
    origin -> the origin of the entire intial scene
    Output: 
    empty_voxel_keys -> indices of the empty voxels (nx3 array)
    non_empty_voxel_keys -> indices of the voxels containing points (nx3 array)
    """
    # if searching for non empty voxels, just return non_empty_voxel_keys
    if return_grid == 'non_empty':
        return non_empty_voxel_keys

    # else continue on finding the empty voxels and return the indices array
    else:
        xmin, ymin, zmin = np.min(non_empty_voxel_keys, axis=0)
        xmax, ymax, zmax = np.max(non_empty_voxel_keys, axis=0)
        
        all_voxel_keys = np.array(np.meshgrid(
                            range(xmin, xmax+1), 
                            range(ymin-1, ymax+2), # add one more row along the view direction
                            range(zmin, zmax+1))
                                ).T.reshape(-1,3) 

        all_voxel_keys, counts = np.unique(np.vstack((all_voxel_keys, non_empty_voxel_keys)), axis=0, return_counts=True)
        if return_grid == 'all':
            return all_voxel_keys

        elif return_grid == 'empty':
            empty_voxel_keys = all_voxel_keys[np.where(counts==1),:][0]
            voxels = len(empty_voxel_keys)

            # filter out voxels that are not between the before and after surfaces
            empty_voxel_keys = utility.filter_empty_space(
                                            empty_voxel_keys, 
                                            (xmin, ymin-1, zmin),
                                            (xmax, ymax+1, zmax)
                                        )

            return empty_voxel_keys


def cluster_grid_space(voxel_keys):
    """
    ::Function to cluster connected voxels
    Input: voxel indices (nx3 array)
    Output: list containing cluster points (list of nx3 arrays)
    """
    keys_pcd = o3d.geometry.PointCloud()
    keys_pcd.points = o3d.utility.Vector3dVector(np.unique(voxel_keys, axis=0))
    labels = np.array(keys_pcd.cluster_dbscan(eps=1.95, min_points=2))
    clusters = [np.asarray(keys_pcd.points)[labels==label] for label in np.unique(labels)]

    return clusters


def export_blocks(block, model1, voxel_size, mode, min_bound, origin, rotation_matrix, out_dir):
    """
    ::Function to export clusters as 3D meshes
    Input: 
    block -> tuple of cluster index (int) and cluster voxels indices (nx3 array)
    voxel_size -> the size of a voxel (float)
    min_bound -> min bound of the entire scene on current view (1x3 array)
    origin -> the origin of the entire intial scene (1x3 array)
    rotation_matrix -> the rotation matrix applied to  the intial scene (3x3 array)
    data_dir -> input/output data directory (str)
    """
    idx, inner_voxels = block

    # surface = o3d.geometry.PointCloud()
    # surface.points = o3d.utility.Vector3dVector(model1)

    surface = o3d.geometry.TriangleMesh()
    surface.vertices = o3d.utility.Vector3dVector(model1[0])
    surface.triangles = o3d.utility.Vector3iVector(model1[1])

    outter_voxels = utility.find_adjacents(inner_voxels)
    volume, error = utility.compute_volume(inner_voxels, outter_voxels, voxel_size)

    if (len(inner_voxels) >= 8) & (len(inner_voxels)/len(outter_voxels)/2 > 1/26):

        # Export 3D object
        mesh = utility.create_3d_object(outter_voxels, voxel_size, min_bound)
        c_type = utility.check_change_type(surface, mesh)

        if c_type == mode:
            R = utility.get_reverse_rotation_matrix(rotation_matrix)  # rotate back to intial view
            mesh = mesh.rotate(R, center=origin)

            cluster_name = "{id:05d}-(v={volume:0.4f} - {error:0.4f}).ply".format(id=idx, volume=volume, error=error)
            o3d.io.write_triangle_mesh(os.path.join(out_dir, cluster_name), mesh)

            shape = utility.get_shape(mesh)
            centroid = mesh.get_center()

            block_dict = {'idx': idx,
                             'volume': volume,
                             'shape': shape,
                             'centroid': centroid}

            return block_dict
        

class RockfallReport():

    def __init__(self, data_dir):

        self.site_name = os.path.split(data_dir)[-1]
        self.headers = ['rockfall ID', 'volume (m3)',
                'A (m)', 'B (m)', 'C (m)',
                'X (m)', 'Y (m)', 'Z (m)']
        self.report = xlsxwriter.Workbook("{}/{}_VoxFall_report.xlsx".format(data_dir, self.site_name))


    def append(self, iter, period):

        def create_datasheet(workbook, site_name, period, headers, data):
            name = "".join(period.split(site_name + '_'))
            worksheet = workbook.add_worksheet(name)

            for i, header in enumerate(headers):
                worksheet.write(0, i, str(header).capitalize())

            for i, entry in enumerate(data):
                for j, header in enumerate(headers):
                    worksheet.write(i+1, j, entry[header])

        rockfalls = list(filter(partial(is_not, None), iter))
        
        if len(rockfalls) > 0:
            data = []
            for rockfall in rockfalls:
                data.append({
                    'rockfall ID': rockfall['idx'],
                    'volume (m3)': rockfall['volume'],
                    'A (m)': rockfall['shape'][0],
                    'B (m)': rockfall['shape'][1],
                    'C (m)': rockfall['shape'][2],
                    'X (m)': rockfall['centroid'][0],
                    'Y (m)': rockfall['centroid'][1],
                    'Z (m)': rockfall['centroid'][2],
                })
            
            create_datasheet(self.report, self.site_name, period, self.headers, data)
    

    def close(self):
        self.report.close()











