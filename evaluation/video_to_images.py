import cv2
import os
import argparse
import subprocess

class V2I():
    @staticmethod
    def upsample_images_in_directory(directory):
        """
        遍历给定目录下所有png图片，使用OpenCV最优方案上采样到1920x1080
        """
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
    
            if not (os.path.isfile(file_path) and filename.lower().endswith('.png')):
                print(f"文件{file_path}不是png图片，跳过")
                continue
    
            # 读取图片 (保留透明度通道)
            img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"无法读取图片：{file_path}")
                continue
    
            # 获取当前分辨率 (注意OpenCV的shape顺序为[height, width])
            h, w = img.shape[:2]
    
            # 检查分辨率是否符合要求
            if (w, h) == (1920, 1080):
                continue
            # ========== 核心优化处理 ==========
            # 分离颜色和透明度通道 (如果有)
            if img.ndim == 3 and img.shape[2] == 4:
                b, g, r, a = cv2.split(img)
                color_img = cv2.merge([b, g, r])
                alpha_channel = a
            else:
                color_img = img
                alpha_channel = None
            # 对颜色通道使用最高质量插值
            upscaled_color = cv2.resize(
                color_img,
                (1920, 1080),
                interpolation=cv2.INTER_LANCZOS4  # 8x8像素邻域插值
            )
            # 对透明度通道使用双三次插值 (减少锯齿)
            if alpha_channel is not None:
                upscaled_alpha = cv2.resize(
                    alpha_channel,
                    (1920, 1080),
                    interpolation=cv2.INTER_CUBIC  # 4x4像素邻域
                )
                upscaled_img = cv2.merge([
                    upscaled_color[..., 0],
                    upscaled_color[..., 1],
                    upscaled_color[..., 2],
                    upscaled_alpha
                ])
            else:
                upscaled_img = upscaled_color
            cv2.imwrite(
                file_path,
                upscaled_img,
                [cv2.IMWRITE_PNG_COMPRESSION, 0]  # 最高质量压缩
            )
            print(f"上采样（至1920x1080）完成：{file_path}")
    
    @staticmethod
    def video_to_frames(video_path, output_dir, frame_prefix="img", frame_format="png"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"无法打开视频文件: {video_path}")
            return
    
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"视频信息: {total_frames} 帧, {fps} FPS, 分辨率: {width}x{height}")
    
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_filename = os.path.join(output_dir, f"{frame_prefix}_{frame_count:05d}.{frame_format}")
            cv2.imwrite(frame_filename, frame)
            frame_count += 1
            # if frame_count % 100 == 0:
            #     print(f"已处理 {frame_count}/{total_frames} 帧")
    
        cap.release()
        # print(f"已处理 {total_frames}/{total_frames} 帧")
        print(f"{video_path}视频分解完成，共保存 {frame_count} 帧图片到 {output_dir}")
        
    @staticmethod
    def ensure_path_exists(path):
        if not os.path.exists(path):
            print("目录不存在，正在为你创建指定目录。。。")
            os.makedirs(path, exist_ok=True)
    
    @staticmethod
    def delet_path(path):
        subprocess.run(["rm", "-rf",path])

