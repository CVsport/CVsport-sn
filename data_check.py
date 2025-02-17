import os
import csv
from PIL import Image
import numpy as np
"""
def images_list_generator(images_path):
    images_list = []
    depth_list = []
    games_list = [os.path.join(images_path, game) for game in os.listdir(images_path)]
    for game in games_list:
        dirs = [os.path.join(game, path) for path in os.listdir(game)]
        for dir in dirs:
            if os.path.isdir(dir):
                data = [os.path.join(dir, path) for path in os.listdir(dir)]
                for d in data:
                    if 'color' in d:
                        images_list = [os.path.join(d, path) for path in os.listdir(d) if '_' not in path]
                    elif 'depth' == d.split('/')[-1]:
                        depth_list = [os.path.join(d, path) for path in os.listdir(d) if '_' not in path]
                    else:
                        continue
                if len(images_list) != len(depth_list):
                    print(data)
    return images_list, depth_list

#images_list_generator("/public_datasets/SoccerNet/football/train/Train")
#images_list_generator("/public_datasets/SoccerNet/football/test/Test")
#images_list_generator("/public_datasets/SoccerNet/football/val/Validation")

# 假设所有CSV文件都在一个文件夹中
folder_path = '/public_datasets/SoccerNet/football/test/Test/game_41/video_1/depth buffer'

# 存储所有深度值的列表
depth_values = []

# 遍历文件夹中的所有CSV文件
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            for i,row in enumerate(csvreader):
                if i == 0:
                    continue
                depth_value = float(row[0])  # 假设深度值在第一列
                depth_values.append(depth_value)

# 找到最大深度和最小深度
max_depth = max(depth_values)
min_depth = min(depth_values)

print(f'Maximum depth: {max_depth}')
print(f'Minimum depth: {min_depth}')
"""
# 读取PNG格式的深度图像
depth_image_path = '/public_datasets/SoccerNet/football/train/Train/game_12/video_1/depth'
images_list = [os.path.join(depth_image_path,path) for path in os.listdir(depth_image_path)]
max_depth = 0
min_depth = 100
image_idx = [0,1]
for image in images_list:
    depth_image = Image.open(image)

    # 将图像转换为NumPy数组
    depth_array = np.array(depth_image, dtype=np.float32)
    depth = depth_array[:,:,:3]
    # 找到最大深度和最小深度
    if depth.max() > max_depth:
        max_depth = np.max(depth)
        idx = np.argmax(depth)
        image_idx[0] = image.split('/')[-1]
    if depth.min() < min_depth:
        min_depth = np.min(depth)
        image_idx[1] = image.split('/')[-1]

print(f'Maximum depth: {max_depth,image_idx[0]}')
print(f'Minimum depth: {min_depth,image_idx[1]}')