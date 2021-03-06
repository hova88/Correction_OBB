import numpy as np
import open3d as o3d

from .src import create_coordinate, create_box_with_arrow ,get_pts_in_box3d

# np.random.randint(0,128,(30,3))  / 255
class PointCloudVis:
    def __init__(self):
        self.points_colorbar = (
            np.array(
                [
                    [127, 127, 127], #defualt
                    [248, 215, 163],  # F8D7A3
                    [143, 98, 63],  # 9d9d9d
                    [85, 109, 170],  # 556DAA
                    [89, 166, 160],  # 59A6A0
                    [84, 172, 83],  # 54AC53
                    [120, 180, 75],  # 78B44B
                    [153, 156, 99],  # 999C63
                    [254, 183, 1],  # FEB701
                ]
            )
            / 255
        )
        
        self.boxes_colorbar = (
            np.array(
                [
                    [255, 0, 0], #defualt
                    [163, 196, 248],  # A3C4F8
                    [248, 7, 11],  # F8070B
                    [170, 146, 85],  # aa9255
                    [166, 89, 95],  # A6595F
                    [171, 83, 172],  # AB53AC
                    [135, 75, 180],  # 874BB4
                    [102, 99, 156],  # 66639C
                    [183, 1, 254],  # B701FE
                ]
            )
            / 255
        )
            
        self.points_colorbar  = np.tile(self.points_colorbar,(1000,1))
        self.boxes_colorbar  = np.tile(self.boxes_colorbar,(1000,1))
    
    def draw_clouds_all_in_one_frame(self, clouds_list , draw_pcd = False):
        """
		clouds_list [ numpy array（N,3/4) ,...] : 
			N: number of points per cloud
		  3/4: number of dims per point 
		"""
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        # --------------------------------------------------------------
        # add point cloud with color
        # --------------------------------------------------------------         
        if not draw_pcd: 
            pc = o3d.geometry.PointCloud()
            points = np.zeros((1, 3))
            colors = [[1, 0, 0]]  # red  
            for i in range(len(clouds_list)):
                cloud = clouds_list[i][:,:3]
                colors += [self.points_colorbar[i]] * cloud.shape[0]
                points = np.vstack((points, cloud))
            pc.points = o3d.utility.Vector3dVector(points)
            pc.colors = o3d.utility.Vector3dVector(colors)
            vis.add_geometry(pc)
        else:
            for pcd in clouds_list:
                pcd.points = pcd.points[:-4]
                vis.add_geometry(pcd)
        # --------------------------------------------------------------
        # coordinate frame
        # --------------------------------------------------------------
        coordinate_frame = create_coordinate(size=2.0, origin=[0, 0, 0])
        vis.add_geometry(coordinate_frame)
        
        # --------------------------------------------------------------
        # drop the window
        # --------------------------------------------------------------
        vis.get_render_option().point_size = 2
        vis.run()
        vis.destroy_window()

    def draw_clouds_one_by_one_frame(self, clouds_list , draw_pcd = False):
        """
		clouds_list [ numpy array（N,3/4) ,...] : 
			N: number of points per cloud
		  3/4: number of dims per point 
		"""
        for i in range(len(clouds_list)):
            vis = o3d.visualization.Visualizer()
            vis.create_window()
            # --------------------------------------------------------------
            # add point cloud with color
            # --------------------------------------------------------------  
            if not draw_pcd: 
                pc = o3d.geometry.PointCloud()
                cloud = clouds_list[i][:,:3]
                colors = [self.points_colorbar[i]] * cloud.shape[0]
                pc.points = o3d.utility.Vector3dVector(cloud)
                pc.colors = o3d.utility.Vector3dVector(colors)
                vis.add_geometry(pc)
            else:
                pcd = clouds_list[i]
                pcd.points = pcd.points[:-4]
                vis.add_geometry(pcd)
            # --------------------------------------------------------------
            # coordinate frame
            # --------------------------------------------------------------
            coordinate_frame = create_coordinate(size=2.0, origin=[0, 0, 0])
            vis.add_geometry(coordinate_frame)
            
            # --------------------------------------------------------------
            # drop the window
            # --------------------------------------------------------------
            vis.get_render_option().point_size = 2
            vis.run()
            vis.destroy_window()

    def draw_clouds_with_boxes_all_in_one_frame(self, clouds_list, boxes_list , draw_pcd = False):
        """
        clouds: [(N, 4) or (N, 3)] [x, y, z, intensity]
        boxes: [(n,7)] np.array = n*7  ( x, y, z, dx, dy, dz, yaw) ##(h,w,l,x,y,z,yaw) 
        """
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        # --------------------------------------------------------------
        # create point cloud
        # --------------------------------------------------------------
        if not draw_pcd:
            clouds = clouds_list[0][:, :3]
            
            point_color = [[140 / 255, 139 / 255, 139 / 255]] * clouds.shape[0]
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(clouds)
            pc.colors = o3d.utility.Vector3dVector(point_color)
            vis.add_geometry(pc)
        else:
            pcd = clouds_list[0]
            pcd.points = pcd.points[:-4]
            vis.add_geometry(pcd)
        # --------------------------------------------------------------
        # create boxes with colors with arrow
        # --------------------------------------------------------------
        boxes_o3d = []
        for i in range(len(boxes_list)):
            boxes = boxes_list[i]
            cloud = clouds_list[i][:, :3]
            cur_box_color = self.boxes_colorbar[-i]

            # --------------------------------------------------------------
            # create points in boxes with same color
            # --------------------------------------------------------------        
            pts_in_box3d = get_pts_in_box3d(cloud , boxes)
            pts_in_box3d_color = [self.boxes_colorbar[-i]] * pts_in_box3d.shape[0]
            

            pc_in_box3d = o3d.geometry.PointCloud()
            pc_in_box3d.points = o3d.utility.Vector3dVector(pts_in_box3d)
            pc_in_box3d.colors = o3d.utility.Vector3dVector(pts_in_box3d_color)

            vis.add_geometry(pc_in_box3d)
            
            for box in boxes: # create boxes
                
                box_o3d, arrow = create_box_with_arrow(box, cur_box_color)
                boxes_o3d.append(box_o3d)
                boxes_o3d.append(arrow)
        
        [vis.add_geometry(element) for element in boxes_o3d] # add_geometry fro boxes

        # --------------------------------------------------------------
        # coordinate frame
        # --------------------------------------------------------------
        coordinate_frame = create_coordinate(size=2.0, origin=[0, 0, 0])
        vis.add_geometry(coordinate_frame)

        # --------------------------------------------------------------
        # drop the window
        # --------------------------------------------------------------
        vis.get_render_option().point_size = 2
        vis.run()
        vis.destroy_window()

    def draw_clouds_with_boxes_one_by_one_frame(self, clouds_list, boxes_list , draw_pcd = False):
        """
        clouds: [(N, 4) or (N, 3)] [x, y, z, intensity]
        boxes: [(n,7)] np.array = n*7  ( x, y, z, dx, dy, dz, yaw) ##(h,w,l,x,y,z,yaw) 
        """
        # assert len(clouds_list)  == len(boxes_list)

        for ind in range(len(clouds_list)):
            vis = o3d.visualization.Visualizer()
            vis.create_window()
            # --------------------------------------------------------------
            # create point cloud
            # --------------------------------------------------------------
            if not draw_pcd:
                clouds = clouds_list[ind][:, :3]
                cur_point_color = [self.points_colorbar[ind]] * clouds.shape[0]
                pc = o3d.geometry.PointCloud()
                pc.points = o3d.utility.Vector3dVector(clouds)
                pc.colors = o3d.utility.Vector3dVector(cur_point_color)
                vis.add_geometry(pc)
            else:
                pcd = clouds_list[ind]
                pcd.points = pcd.points[:-4]
                vis.add_geometry(pcd)
            # --------------------------------------------------------------
            # create boxes with colors with arrow
            # --------------------------------------------------------------
            boxes_o3d = []
            boxes = boxes_list[ind]
            cur_box_color = self.boxes_colorbar[ind]
            # create boxes
            for box in boxes:
                box_o3d, arrow = create_box_with_arrow(box, cur_box_color)
                boxes_o3d.append(box_o3d)
                boxes_o3d.append(arrow)
            # add_geometry fro boxes
            [vis.add_geometry(element) for element in boxes_o3d]

            # --------------------------------------------------------------
            # coordinate frame
            # --------------------------------------------------------------
            coordinate_frame = create_coordinate(size=2.0, origin=[0, 0, 0])
            vis.add_geometry(coordinate_frame)

            # --------------------------------------------------------------
            # drop the window
            # --------------------------------------------------------------
            vis.get_render_option().point_size = 2
            vis.run()
            vis.destroy_window()

    def draw_clouds_with_boxes_gt_and_pred(self, clouds_list, boxes_list , draw_pcd = False):
        """
        clouds: [(N, 4) or (N, 3)] [x, y, z, intensity]
        boxes: [(n,7)] np.array = n*7  ( x, y, z, dx, dy, dz, yaw) ##(h,w,l,x,y,z,yaw) 
        """
        # assert len(clouds_list)  == len(boxes_list)

        
        for ind in range(len(clouds_list)):
            vis = o3d.visualization.Visualizer()
            vis.create_window()
            # --------------------------------------------------------------
            # create point cloud
            # --------------------------------------------------------------
            if not draw_pcd:
                clouds = clouds_list[ind][:, :3]
                print("index is :" , ind)
                print(clouds.shape)
                cur_point_color = [[0, 0, 0]] * clouds.shape[0]
                pc = o3d.geometry.PointCloud()
                pc.points = o3d.utility.Vector3dVector(clouds)
                pc.colors = o3d.utility.Vector3dVector(cur_point_color)
                vis.add_geometry(pc)

                # coords = coords_list[:, :3]
                # # print("index is :" , ind)
                # # print(coords.shape)
                # cur_point_color = [[255, 0, 0]] * coords.shape[0]
                # pc = o3d.geometry.PointCloud()
                # pc.points = o3d.utility.Vector3dVector(coords)
                # pc.colors = o3d.utility.Vector3dVector(cur_point_color)
                # vis.add_geometry(pc)

            else:
                pcd = clouds_list[ind]
                pcd.points = pcd.points[:-4]
                vis.add_geometry(pcd)
            # --------------------------------------------------------------
            # create boxes with colors with arrow
            # --------------------------------------------------------------
            boxes_o3d = []
            boxes = boxes_list[ind]
            gt_boxes = boxes[0]
            fake_boxes = boxes[1]
            pred_boxes = boxes[2]
            
            
            
            gt_box_color =[255, 0, 0]
            fake_box_color =[ 0,255, 0]
            pred_box_color = [0, 0, 255]      
            # create boxes
            for gt_box in gt_boxes:
                box_o3d, arrow = create_box_with_arrow(gt_box, gt_box_color)
                boxes_o3d.append(box_o3d)
                # boxes_o3d.append(arrow)

            for fake_box in fake_boxes:
                box_o3d, arrow = create_box_with_arrow(fake_box, fake_box_color)
                boxes_o3d.append(box_o3d)
                # boxes_o3d.append(arrow)

            for pred_box in pred_boxes:
                box_o3d, arrow = create_box_with_arrow(pred_box, pred_box_color)
                boxes_o3d.append(box_o3d)
                # boxes_o3d.append(arrow)

            # add_geometry fro boxes
            [vis.add_geometry(element) for element in boxes_o3d]

            # --------------------------------------------------------------
            # coordinate frame
            # --------------------------------------------------------------
            coordinate_frame = create_coordinate(size=2.0, origin=[0, 0, 0])
            vis.add_geometry(coordinate_frame)

            # --------------------------------------------------------------
            # drop the window
            # --------------------------------------------------------------
            vis.get_render_option().point_size = 3
            vis.run()
            vis.destroy_window()
