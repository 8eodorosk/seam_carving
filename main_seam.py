import sys
from imageio import imread, imwrite
from Seam import Carve, CarveFirstSeam
import numpy as np


filter_du_sobel = np.array([
        [1.0, 2.0, 1.0],
        [0.0, 0.0, 0.0],
        [-1.0, -2.0, -1.0],
    ])

filter_dv_sobel = np.array([
        [1.0, 0.0, -1.0],
        [2.0, 0.0, -2.0],
        [1.0, 0.0, -1.0],
    ])

filter_du_scharr = np.array([
        [3.0, 10.0, 3.0],
        [0.0, 0.0, 0.0],
        [-3.0, -10.0, -3.0],
    ])

filter_dv_scharr = np.array([
        [3.0, 0.0, -3.0],
        [10.0, 0.0, -10.0],
        [3.0, 0.0, -3.0],
    ])

filter_du_prewitt = np.array([
        [1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0],
        [-1.0, -1.0, -1.0],
    ])

filter_dv_prewitt = np.array([
        [1.0, 0.0, -1.0],
        [1.0, 0.0, -1.0],
        [1.0, 0.0, -1.0],
    ])

def main():
    which_axis = sys.argv[1]
    # twn arithmo twn pixel pou theloume na kopsoume
    scale = int(sys.argv[2])

    # tin eikona pou tha diabasoume
    in_filename = sys.argv[3]

    # tin eikona pou tha vgalei
    out_filename = which_axis + "_" + str(scale) + sys.argv[4] + ".jpg"

    # ton operator pou tha xrisimpoihsoume
    type = sys.argv[5]
    # type = "sobel"

    if (type == "sobel"):
        filter_du = filter_du_sobel
        filter_dv = filter_dv_sobel
    elif (type == "scharr"):
        filter_du = filter_du_scharr
        filter_dv = filter_dv_scharr
    elif (type == "prewitt"):
        filter_du = filter_du_prewitt
        filter_dv = filter_dv_prewitt

    img = imread(in_filename)

    x = CarveFirstSeam(filter_du, filter_dv, img, scale)

    if which_axis == 'showSeamHeight':
        out = x.showSeamHeight()
    elif which_axis == 'showSeamWidth':
        out = x.showSeamWidth()

    imwrite(out_filename, out)

if __name__ == '__main__':
    main()