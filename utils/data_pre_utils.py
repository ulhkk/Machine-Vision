import os
import math
import numpy as np
#import itertools
#import open3d as o3d
# import pandas as pd
# from tqdm import tqdm
# import joblib
# import time
import rosbag
import sensor_msgs.point_cloud2 as pc2
import torch
import yaml
'''
- 
  name: "x"
  offset: 0
  datatype: 7
  count: 1
- 
  name: "y"
  offset: 4
  datatype: 7
  count: 1
- 
  name: "z"
  offset: 8
  datatype: 7
  count: 1
- 
  name: "intensity"
  offset: 16
  datatype: 7
  count: 1
- 
  name: "t"
  offset: 20
  datatype: 6
  count: 1
- 
  name: "reflectivity"
  offset: 24
  datatype: 4
  count: 1
- 
  name: "ring"
  offset: 26
  datatype: 2
  count: 1
- 
  name: "noise"
  offset: 28
  datatype: 4
  count: 1
- 
  name: "range"
  offset: 32
  datatype: 6
  count: 1
---
'''

#FILENAME = '/home/hanli/Documents/waymo/segment-15533468984793020049_800_000_820_000_with_camera_labels.tfrecord'
labelMapping = {
    "0": "Pedestrian",
    "1": "Cyclist",
    "2": "Car",
    "3": "Motorcycle",
    "0": "Truck",
    "0": "Pedestrian",
    "0": "Pedestrian",
}


def reshape_torch(pointcloud, num_field):
    device = torch.device("cpu")
    pointcloud2 = torch.tensor(pointcloud, dtype=torch.float32, device=device)
    return pointcloud2.reshape(
        (-1, num_field)).detach().numpy().astype(np.float32)


def euler_from_quaternion(x, y, z, w):
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z  #- math.pi/2# in radians


def obj_to_quad(obj):
    l_x = obj.dimensions.x
    l_y = obj.dimensions.y
    x_0 = obj.pose.position.x
    y_0 = obj.pose.position.y

    w = obj.pose.orientation.w
    x = obj.pose.orientation.x
    y = obj.pose.orientation.y
    z = obj.pose.orientation.z

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    A_x = l_x / 2 * math.cos(yaw) - l_y / 2 * math.sin(yaw) + x_0
    A_y = l_x / 2 * math.sin(yaw) + l_y / 2 * math.cos(yaw) + y_0

    B_x = -l_x / 2 * math.cos(yaw) - l_y / 2 * math.sin(yaw) + x_0
    B_y = -l_x / 2 * math.sin(yaw) + l_y / 2 * math.cos(yaw) + y_0

    C_x = -l_x / 2 * math.cos(yaw) + l_y / 2 * math.sin(yaw) + x_0
    C_y = -l_x / 2 * math.sin(yaw) - l_y / 2 * math.cos(yaw) + y_0

    D_x = l_x / 2 * math.cos(yaw) + l_y / 2 * math.sin(yaw) + x_0
    D_y = l_x / 2 * math.sin(yaw) - l_y / 2 * math.cos(yaw) + y_0

    return A_x, A_y, B_x, B_y, C_x, C_y, D_x, D_y, yaw


def in_obj(point, obj):
    A_x, A_y, B_x, B_y, C_x, C_y, D_x, D_y, yaw = obj_to_quad(obj)

    if point[0] > obj.pose.position.x + 20:
        return False, yaw
    if point[1] > obj.pose.position.y + 20:
        return False, yaw
    if point[2] < (obj.pose.position.z - obj.dimensions.z / 2 + 0.1):
        return False, yaw

    a = (B_x - A_x) * (point[1] - A_y) - (B_y - A_y) * (point[0] - A_x)
    b = (C_x - B_x) * (point[1] - B_y) - (C_y - B_y) * (point[0] - B_x)
    c = (D_x - C_x) * (point[1] - C_y) - (D_y - C_y) * (point[0] - C_x)
    d = (A_x - D_x) * (point[1] - D_y) - (A_y - D_y) * (point[0] - D_x)

    if (a > 0 and b > 0 and c > 0 and d > 0) or (a < 0 and b < 0 and c < 0
                                                 and d < 0):
        return True, yaw
    return False, yaw


def ProcessRosbag():
    base_path = "/media/sikun/ld_harddisk/sikun/LD_compass/dataset/org"
    save_path = '/media/sikun/ld_harddisk/sikun/LD_compass/dataset/data/'
    save_label_path = '/media/sikun/ld_harddisk/sikun/LD_compass/dataset/label/'

    bag_PATH = base_path + '/split_ouster_128_20210608141208_004_processed_sync_00_OD.bag'

    save_validation_path = save_path + '/validation'

    bag_num = 0
    index = 0
    topic_name = ['/ld_object_lists', '/os_cloud_node/points']

    object_lists = []

    try:
        with rosbag.Bag(bag_PATH, 'r') as bag:
            for topic, msg, t in bag.read_messages(topics='/ld_object_lists'):
                # for obj in msg.objects:
                object_lists.append(msg.objects)
                # print(len(msg.objects))
    finally:
        bag.close()

    try:
        with rosbag.Bag(bag_PATH, 'r') as bag:
            for topic, msg, t in bag.read_messages(topics=topic_name):
                if topic == str('/os_cloud_node/points'):
                    for i, obj in enumerate(object_lists[index]):
                        lidar = pc2.read_points(msg, skip_nans=True)
                        index_num = str(bag_num).zfill(4) + "_" + str(
                            index).zfill(4) + "_" + str(i).zfill(4)
                        bin_file = save_path + index_num + ".bin"
                        yaml_file = save_label_path + index_num + ".yaml"

                        point_arry = []
                        for point in lidar:
                            in_box, yaw = in_obj(point, obj)
                            if in_box:
                                point_arry.append(
                                    [point[0], point[1], point[2], point[3]])
                                # print("point: ", point, " is in ", index_num)

                        obj_yaml = {
                            'Yaw': yaw,
                            'Dim_x': obj.dimensions.x,
                            'Dim_y': obj.dimensions.y,
                            'Dim_z': obj.dimensions.z,
                            'Pose_x': obj.pose.position.x,
                            'Pose_y': obj.pose.position.y,
                            'Pose_z': obj.pose.position.z,
                            'Velocity': obj.velocity.linear.x,
                            'class_name': obj.class_label_true,
                            'Num_of_points': len(point_arry),
                            'Difficulty': 0
                        }

                        with open(yaml_file, "w", encoding="utf8") as f:
                            yaml.dump(obj_yaml, f)
                        np.array(point_arry).astype(
                            np.float32).tofile(bin_file)
                        # print("===============================")
                    index += 1
                    print(index)
            # for topic, msg, t in bag.read_messages(topics='/ld_object_lists'):
            # for obj in msg.objects:
            # object_lists.append(msg.objects)

            # new_label.append(obj.dimensions.z)
            # new_label.append(obj.dimensions.y + 0.2)
            # new_label.append(obj.dimensions.x + 0.4)
            # new_label.append(obj.pose.position.x)
            # new_label.append(obj.pose.position.y)
            # new_label.append(obj.pose.position.z - 1.74)

            # roll_x, pitch_y, yaw_z = euler_from_quaternion(
            #     obj.pose.orientation.x, obj.pose.orientation.y,
            #     obj.pose.orientation.z, obj.pose.orientation.w)
            # print(index)

    finally:
        bag.close()


"""
for number in range(len(dirs)):

    filename = dirs[number]
    dirs_s = os.listdir(save_path+'/'+filename)
    
    for filename_s in tqdm(dirs_s):
        
        dataset = tf.data.TFRecordDataset(save_path + '/' + filename + '/' + filename_s, compression_type='')
        
        for data in dataset:
            
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
        
        
            (range_images, camera_projections,
            range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(
                frame)
        
        
            points= frame_utils.convert_range_image_to_point_cloud(
                frame,
                range_images,
                camera_projections,
                range_image_top_pose)
            
            #points_all = np.concatenate(points, axis=0)
            print(points.shape)
            points[:,2] =points[:,2] - 1.73
            points=points[(points[:,3]<1) ,:] # filter intensity > 1
            points.tofile(save_bin_path+str(i).zfill(6)+".bin")
            
            
            
            df = pd.DataFrame(columns=['type','truncated','occluded','alpha','a','b','c','d','height','width','length','x','y','z','yaw'],data=[])
    
            
            for label in frame.laser_labels:
                new_label = []
                hwl = np.zeros((1,3))
                hwl[0,0] = label.box.height -0.14
                hwl[0,1] = label.box.width-0.13
                hwl[0,2] = label.box.length-0.15
                
                
                if label.type == 0:
                    continue
                else:
                    if label.type == 1:
                    new_predict = _map[str(int(clf.predict(hwl)[0]))] # new label
                    #print(new_predict)
                    new_label.append(new_predict)
                    
                    if new_predict == 'Car':
                        height_car.append(label.box.center_z-1.8)

                    if new_predict == 'Van':
                        height_van.append(label.box.center_z-1.8)  

                    if new_predict == 'Truck':
                        height_truck.append(label.box.center_z-1.8)  
                            
                    elif label.type == 2:
                        new_label.append('Pedestrian')
                        height_people.append(label.box.center_z-1.8)

                    elif label.type == 3:
                        print('traffic sign')
                        new_label.append('Sign')
                        height_sign.append(label.box.center_z-1.8)

                    elif label.type == 4:

                        new_label.append('Cyclist')
                        height_cyc.append(label.box.center_z-1.8)
                    print(label.detection_difficulty_level)
                    print(label.tracking_difficulty_level)

                    #break
                    new_label.append(label.tracking_difficulty_level)
                    new_label.append(label.detection_difficulty_level)
                    new_label.append(0)
                    new_label.append(0)
                    new_label.append(0)
                    new_label.append(0)
                    new_label.append(0)
                    
                    if label.type == 1:
                        
                        new_label.append(label.box.height-0.14)
                        new_label.append(label.box.width-0.13)
                        new_label.append(label.box.length-0.17)
                    else:
                        new_label.append(label.box.height)
                        new_label.append(label.box.width-0.08)
                        new_label.append(label.box.length-0.09)
                        
                    new_label.append(label.box.center_x)
                    new_label.append(label.box.center_y)
                    new_label.append(label.box.center_z-1.73)
                    new_label.append(label.box.heading)
                    
                    df.loc[df.shape[0]+1]=new_label
                    #print(temp_label)
        
            df.to_csv(save_label_path+str(i).zfill(6)+".txt",sep=' ', index=False, header=False)    
        
            #points_all = np.concatenate(points, axis=0)
    
            i += 1
            print(i)

print(f'car of z : {np.mean(height_car)}')
print(f'truck of z : {np.mean(height_truck)}')
print(f'van of z : {np.mean(height_van)}')
print(f'people of z : {np.mean(height_people)}')
print(f'cyc of z : {np.mean(height_cyc)}')
print(f'sign of z : {np.mean(height_sign)}')
"""
#%%
if __name__ == '__main__':
    ProcessRosbag()

# %%