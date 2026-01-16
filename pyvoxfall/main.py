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

import os
import numpy as np
import time
import multiprocessing.dummy as mp
import core
import copy
import xlsxwriter
from operator import is_not
from functools import partial


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


def main(voxel_size, direction, mode, data_dir, iter, header):

    """ 
    PART 1: DATA IMPORT 
    """ 
    start = time.time()
    print("::Loading Data")
    np.random.seed(1234)

    mesh1, mesh2 = core.import_triangle_mesh_data(data_dir, iter)
    # mesh1, mesh2 = core.import_point_cloud_data(data_dir, iter, noise=0.02)
    mesh1, mesh2, R, origin = core.align_with_xz_view(mesh1, mesh2, direction)
    mesh = copy.deepcopy(mesh1)
    mesh += mesh2

    end = time.time()
    print("  Elapsed time: {:0.2f} sec".format(end - start))


    """ 
    PART 2: PREPROCESSING 
    """
    start = time.time()
    print("::Preprocessing")

    windows = core.crop_windows(mesh, voxel_size, size=3)

    end = time.time()
    print("  Elapsed time: {:0.2f} sec".format(end - start))


    """
    PART 3: PARALLEL PROCESSING TO FIT OCCUPANCY GRID
    """
    start = time.time()
    print("::Fitting Occupancy Grid")

    pool = mp.Pool()
    
    fit_occupancy_grid = partial(core.fit_occupancy_grid,
                                return_grid='empty')

    empty_voxel_keys = pool.map(fit_occupancy_grid, windows)

    end = time.time()
    print("  Elapsed time: {:0.2f} sec".format(end - start))


    """ 
    PART 4: CLUSTER EMPTY SPACE AND EXPORT DETECTIONS
    """
    start = time.time()
    print("::Clustering Empty Space")
        
    out_dir = os.path.join(data_dir, 'output', header)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    blocks = core.cluster_grid_space(np.vstack(empty_voxel_keys))
    
    export_blocks = partial(core.export_blocks,
                            model1=[np.asarray(mesh1.vertices),np.asarray(mesh1.triangles)],
                            # model1=np.asarray(mesh1.points),
                            voxel_size=voxel_size,
                            mode=mode,
                            min_bound=mesh.get_min_bound(),
                            origin=origin,
                            rotation_matrix=R,
                            out_dir=out_dir)

    block_dict = pool.map(export_blocks, enumerate(blocks))

    pool.close()

    end = time.time()
    print("  Elapsed time: {:0.2f} sec".format(end - start))

    return block_dict











