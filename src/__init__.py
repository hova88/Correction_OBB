import numpy as np

def ball_query(points, points_in_fb , r = 0.15 , leaf_size = 2):
    from sklearn.neighbors import BallTree
    """
    Args:
        points:(N,3) [x,y,z] 原始点云
        points_in_fb: (M, 3) [x, y, z] fake_box内的点云
        r: 搜索半径
        leaf_size: 叉树最小叶子尺寸
    Returns:
        points_from_bq: (Q, 3) [x,y,z] 
    """
    tree = BallTree(points, leaf_size=leaf_size) 
    ind = tree.query_radius(points_in_fb, r=r)
    ind_list = []
    for i in ind:
        ind_list += i.tolist()
    return points[ind_list]  

def get_obb_from_points(points):
    from pyobb.obb import OBB
    """
    Args:
        points:(N,3) [x,y,z] 
    Returns:
        correct_box: (7,) [x_hat, y_hat, z_hat, dx_hat, dy_hat, dz_hat, yaw_hat] 
    """
    obb = OBB.build_from_points(points)
    x_hat, y_hat, z_hat = obb.centroid[0] , obb.centroid[1] , obb.centroid[2]
    dx_hat, dy_hat, dz_hat = obb.extents[0]*2 , obb.extents[1]*2 , obb.extents[2]*2
    yaw_hat = np.arctan2(obb.rotation[0,2] , obb.rotation[2,2]) + np.pi/2
    return np.array([x_hat, y_hat, z_hat, dx_hat, dy_hat, dz_hat, yaw_hat])

def get_pred_box(points,
                 points_in_fb,
                 fk_boxes,
                 min_points_for_crop = 50, 
                 crop_lower_bound = 0.3, 
                 ball_query_radius =0.15):
    """
    Args:
        points: (N,3)
        points_in_fb: (M,3)
        min_points_for_crop: int 
        crop_lower_bound: float
        ball_query_radius: float
    Returns:
        pred_box: (7,) [x,y,z,dx,dy,dz,yaw] 
    """
    if len(points_in_fb) >= min_points_for_crop:
        points_in_fb = points_in_fb[points_in_fb[:,2] > crop_lower_bound]
  
    # get correct paraments
    points_from_bq = ball_query(points , points_in_fb , r = ball_query_radius)
    correct_box = get_obb_from_points(points_from_bq)
 
    x =  (fk_boxes[0,0] + correct_box[0]) /2 
    y =  (fk_boxes[0,1] + correct_box[1]) /2 
    z =   fk_boxes[0,2] # 由于采用了bound_crop的策略，所以不对高度进行修正是一个比较保险的策略
    dx = (fk_boxes[0,3] + correct_box[3]) /2 
    dy = (fk_boxes[0,4] + correct_box[4]) /2 
    dz = fk_boxes[0,5]
    yaw = correct_box[6]
    return np.array([x, y, z, dx, dy, dz, yaw])

if __name__ == "__main__":
    points = np.random.randn(100,3)
    points_in_fb = points[:60]
    fk_boxes = np.array([0,0,0,1,1,1,0]).reshape(1,7)
    print(get_pred_box(points,points_in_fb,fk_boxes))