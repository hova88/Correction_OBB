# Post Detector

## 由于深度学习的方法始终无法获得好的效果，所以在此分支应用**主成分分析法**与**临近点查询**想结合，作为包围盒矫正的一个保底方案
---


##  🤡 算法设计
```bash
#0:  已知：points(N,3):[x,y,z] , fake_boxes(1,7):[x,y,z,dx,dy,dz,yaw] , gt_boxes(1,7):[x,y,z,dx,dy,dz,yaw]
||
|| [->1] 判断points是否在fake_boxes内(默认选取高于0.2的points)
\/
#1:  得到：points_in_fb(M,3): [x,y,z]
||
|| [->2] 运用BALLTREE对 points_in_fb 进行临近点搜索(默认半径为0.15)
\/
#2:  得到：points_from_bq(Q,3):[x,y,z]
||
|| [->3] 运用PCA对 point_from_bq 进行有向包围盒(OBB)的生成
\/
#3:  得到：obb(1,7): {x,y,z,dx,dy,dz,yaw}_hat
||
|| [->4] 对obb与fake_boxes进行融合 , 修正了fake_box的 [x,y,dx,dy,yaw]
\/
#4:  得到：pred_box(1,7): [(x + x_hat) / 2 ,(y + y_hat) / 2 , z ,  (dx + dx_hat) / 2 ,(dy + dy_hat) / 2 , dz , yaw_hat]
```

##  🧩 依赖
```bash 
pip install numpy pyobb sklearn open3d 
python setup.py develop
```

## 📗 使用方法

- 首先需要将数据存成pkl格式，[{'points','fake_boxes','fb_mask','gt_boxes','gt_mask'},...] , 我在`nvidia@172.22.52.12:~/shared_data/lyft_bus`里面存放了一个样板数据，以供参考。
- 运行程序，得到result.pkl
```bash
python main.py --data_path *.pkl
```

## 📊 性能评估
```bash
python tools/analyzer.py
```
采用对矫正前后的**pred_boxes , fake_boxes**分别与真值**gt_boxes**计算IOU,然后求差，统计IOU差值分布情况


<p align="center">
  <img width="800" alt="fig_method" src=docs/stats_result.png>
</p>


 如图所示，横坐标为正的表示矫正起到了作用，反之就是没作用。所以用四个字评价当前的效果： **乏善可陈** 

##  👀 可视化
```bash
python tools/viewer.py
```
选取一些看起来还不错的效果展示一下。`红框`表示真值框(gt_boxes),`蓝框`表示主网络检测出来的结果(fake_boxes),`绿框`表示矫正后的结果(pred_boxes)

<p align="center">
  <img width="800" alt="fig_method" src=docs/ScreenCapture_2021-04-22-17-07-05.png>
</p>
