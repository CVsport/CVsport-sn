import cv2
import numpy as np
import os
import subprocess
import json
def ensure_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def video_generator(frame_rate,frame_number,root,video_name,save_path,temp_path):
    images = os.listdir(root)
    image_paths = [os.path.join(root,image) for image in images if '_' not in image]
    length = len(image_paths)
    if not length == len(os.listdir(root.replace('color','depth'))) == frame_number :
        print(video_name)
        return
    # 定义视频参数
    video_name = video_name + ".mp4"
    output_video_rgb_path = temp_path + video_name  # RGB 视频路径
    output_video_depth_path = temp_path + video_name.replace('color','depth')  # 深度视频路径
    fps = frame_rate  # 视频帧率
    frame_size = (1920,1080)  # 假设图像尺寸是 1920x1080

    # 创建视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 或者尝试使用 'XVID'
    video_writer_rgb = cv2.VideoWriter(output_video_rgb_path, fourcc, fps, frame_size)
    video_writer_depth = cv2.VideoWriter(output_video_depth_path, fourcc, fps, frame_size)

    for idx in range(1,length+1):
        # 获取一个场景
        scene = root + '/' + str(idx) + '.png'  # 获取第一个场景

        # 遍历每一帧，写入视频
        # 获取RGB图像路径
        rgb_image = cv2.imread(scene)
        depth_image = cv2.imread(scene.replace('color', 'depth'), cv2.IMREAD_UNCHANGED)

        # 如果深度图是16位的，可能需要对其进行归一化或者转换为可视化的灰度图
        frame_depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
        frame_depth_normalized = np.uint8(frame_depth_normalized)  # 转换为uint8类型

        # 确保深度图和RGB图像的大小一致
        frame_rgb_resized = cv2.resize(rgb_image, frame_size)
        frame_depth_resized = cv2.resize(frame_depth_normalized, frame_size)

        # 将RGB帧和深度帧写入视频
        video_writer_rgb.write(frame_rgb_resized)
        video_writer_depth.write(frame_depth_resized)

    # 释放视频写入器
    video_writer_rgb.release()
    video_writer_depth.release()

    #将视频移入public_datasets
    subprocess.run(['mv',output_video_rgb_path,save_path + '/color'])
    subprocess.run(['mv',output_video_depth_path,save_path + '/depth'])
    print(f"{video_name} saved successfully.")

def video_data_prepare(data_path,temp_path):
    if not os.path.exists(data_path):
        return
    if os.path.isdir(data_path):
        data_list = [os.path.join(data_path,path) for path in os.listdir(data_path)]
        for item in data_list:
            if os.path.isdir(item):
                dirs = [os.path.join(item,path) for path in os.listdir(item)]
                for dir in dirs:
                    if os.path.isdir(dir) and 'video' in dir:
                        data = [os.path.join(dir,path) for path in os.listdir(dir)]
                        json_path = item + '/' + item.split('/')[-1] + '.json'
                        with open(json_path, 'r') as f:
                            f = json.load(f)
                            print(f)
                            frame_rate = float(f["Frame rate"])
                            frame_number = int(f["Number of frames"])
                        for da in data:
                            images = [os.path.join(da,path) for path in os.listdir(da)]
                            if images[0].endswith('.png') and 'color' in da:
                                video_name = item.split('/')[-1]+'_'+da.split('/')[-1]+'_'+dir.split('/')[-1]
                                video_generator(
                                    frame_rate,
                                    frame_number,
                                    da,
                                    video_name,
                                    data_path.replace(data_path.split('/')[-1],'video'),
                                    temp_path)
# 加载数据集
root = "/public_datasets/SoccerNet/football"
train_path = os.path.join(root, "train","Train")
val_path = os.path.join(root, "val","Validation")
test_path = os.path.join(root, "test","Test")
temp_path = '/home/ipad_3d/dmy/Depth-Anything-V2/video/'
ensure_path(temp_path)

video_data_prepare(train_path,temp_path)
video_data_prepare(val_path,temp_path)
video_data_prepare(test_path,temp_path)

os.remove(temp_path)
