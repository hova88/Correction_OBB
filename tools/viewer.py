import pickle
import numpy as np
from vis_utils.PointCloudVis import PointCloudVis
import open3d as o3d

def rotz(t):
    """ Rotation about the z-axis. """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def dataloader(file_path):
    clouds_list = []
    boxes_list = []

    with open(file_path , 'rb') as f:
        infos = pickle.load(f)
    for info in infos[:10]:
        # info = infos[8]
        points = info['points']        
        boxes = [info['gt_boxes'] , info['pred_boxes'], info['fake_boxes']]

        clouds_list.append(points)
        boxes_list.append(boxes)

    return clouds_list , boxes_list 


if __name__ == "__main__":
    import argparse
    import glob
    from pathlib import Path

    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument(
        "--result_file",
        type=str,
        default="data/result_lyft_bus_20.pkl",
        help="specify the pkl file for view",
    )
    args = parser.parse_args()

    clouds_list , boxes_list = dataloader(args.result_file)
    # print("Done")
    
    V = PointCloudVis()
    V.draw_clouds_with_boxes_gt_and_pred(clouds_list , boxes_list) 

