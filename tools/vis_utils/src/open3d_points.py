import numpy as np
import open3d as o3d
from open3d import geometry
import numpy as np 

def get_pts_in_box3d(points , boxes):
    """
    :param 
        points: (#points,3/4)
        boxes: (N,7)
    :return:
        pt_in_box3d: (n, 3/4)
    """
    pts_in_box3d = np.zeros((1,points.shape[1]))
    for box in boxes:
        cx,cy,cz,dx,dy,dz,rz = box[0],box[1],box[2],box[3],box[4],box[5],box[6]
        
        local_z = points[:,2] - cz
        cosa = np.cos(-rz) 
        sina = np.sin(-rz)
        local_x = (points[:,0] - cx) * cosa + (points[:,1] - cy) * (-sina)
        local_y = (points[:,0] - cx) * sina + (points[:,1] - cy) * cosa
        
        # Finding the intersection 
        condition = (np.abs(local_z)  < dz/2) \
                  * (np.abs(local_y ) < dy/2) \
                  * (np.abs(local_x ) < dx/2) 
        pts_in_box3d = np.vstack((points[condition],pts_in_box3d))

    return pts_in_box3d


