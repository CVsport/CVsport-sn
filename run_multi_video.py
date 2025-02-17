import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch
from depth_anything_v2.dpt import DepthAnythingV2

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2')

    parser.add_argument('--video-path', type=str)
    parser.add_argument('--gt-path', type=str, default=None)
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--outdir', type=str, default='./vis_video_depth')

    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])

    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')

    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    state_dict = torch.load('/data/ipad_3d/dmy/depthanything_output/latest.pth', map_location='cpu')
    state_dict_vitl = torch.load('/home/ipad_3d/dmy/Depth-Anything-V2/checkpoints/depth_anything_v2_vitl.pth',
                                 map_location='cpu')
    new_state_dict = {}
    for key in state_dict.keys():
        if key == 'model':
            for m_key in state_dict[key].keys():
                new_key = m_key.replace("module.", "")
                new_state_dict[new_key] = state_dict[key][m_key]
                if 'depth_head' in m_key:
                    new_state_dict[new_key] = state_dict_vitl[new_key]
        else:
            continue
    depth_anything.load_state_dict(new_state_dict)
    depth_anything = depth_anything.to(DEVICE).eval()

    if os.path.isfile(args.video_path):
        if args.video_path.endswith('txt'):
            with open(args.video_path, 'r') as f:
                lines = f.read().splitlines()
        else:
            filenames = [args.video_path]
    else:
        filenames = glob.glob(os.path.join(args.video_path, '**/*'), recursive=True)

    os.makedirs(args.outdir, exist_ok=True)

    margin_width = 50
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')

    for k, filename in enumerate(filenames):
        print(f'Progress {k + 1}/{len(filenames)}: {filename}')

        raw_video = cv2.VideoCapture(filename)
        if args.gt_path:
            gt_video = cv2.VideoCapture(args.gt_path)
        frame_width, frame_height = 518,291#int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
            #raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_rate = int(raw_video.get(cv2.CAP_PROP_FPS))

        if args.pred_only:
            if args.gt_path:
                output_width = frame_width * 2 + margin_width
            else:
                output_width = frame_width
        else:
            if args.gt_path:
                output_width = frame_width * 3 + margin_width * 2
            else:
                output_width = frame_width * 2 + margin_width

        output_path = os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '.mp4')
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (output_width, frame_height))

        while raw_video.isOpened() and (gt_video.isOpened() if args.gt_path else True):
            ret, raw_frame = raw_video.read()
            if not ret:
                break
            raw_frame = cv2.resize(raw_frame, (frame_width, frame_height))
            if args.gt_path:
                ret2, gt_frame = gt_video.read()
                gt_frame = cv2.resize(gt_frame, (frame_width, frame_height))
                gt_frame = gt_frame[:,:,0]
                if not ret2:
                    break

            depth = depth_anything.infer_image(raw_frame, args.input_size)

            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            depth = depth.astype(np.uint8)

            if args.grayscale:
                depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
            else:
                depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
                if args.gt_path:
                    gt_frame = (cmap(gt_frame)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
            print(depth.shape, raw_frame.shape, gt_frame.shape)
            if args.pred_only:
                if args.gt_path:
                    split_region = np.ones((frame_height, margin_width, 3), dtype=np.uint8) * 255
                    combined_frame = cv2.hconcat([gt_frame, split_region, depth])
                    out.write(combined_frame)
                else:
                    out.write(depth)
            else:
                if args.gt_path:
                    split_region = np.ones((frame_height, margin_width, 3), dtype=np.uint8) * 255
                    combined_frame = cv2.hconcat([raw_frame, split_region, gt_frame, split_region, depth])
                    cv2.imwrite('/home/ipad_3d/dmy/Depth-Anything-V2/demo_output/1.png', combined_frame)
                    out.write(combined_frame)
                else:
                    split_region = np.ones((frame_height, margin_width, 3), dtype=np.uint8) * 255
                    combined_frame = cv2.hconcat([raw_frame, split_region, depth])
                    out.write(combined_frame)

        raw_video.release()
        if args.gt_path:
            gt_video.release()
        out.release()
