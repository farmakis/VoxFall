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

import main
import os
import argparse


# Define input parameters
parser = argparse.ArgumentParser()
parser.add_argument("--project", type=str, default='data', help="Project name")
parser.add_argument("--mode", type=str, default='loss', help="Change type searched", choices=['loss', 'gain'])
parser.add_argument("--res", type=float, default=.05, help="Voxel size")
parser.add_argument("--az", type=int, default=105, help="Slope azimuth")
args = parser.parse_args()


if __name__ == '__main__':

    data_dir = os.path.join(os.getcwd(), args.project)
    in_data = os.listdir(os.path.join(data_dir, 'input'))
    in_data.sort()

    report = main.RockfallReport(data_dir)

    for i in range(len(in_data)-1):

        header = in_data[i].split('.')[0] + '_vs_' + in_data[i+1].split('.')[0]
        print("\n" + header + "\n-----------------------------")

        rockfalls = main.main(
                        voxel_size=args.res,
                        direction=args.az,
                        mode=args.mode,
                        data_dir=data_dir,
                        iter=i,
                        header=header
                        )
        
        report.append(rockfalls, header)

    report.close()
        







