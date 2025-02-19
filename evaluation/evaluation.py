import torch
import argparse
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from video_to_images import V2I


from utils import RunningAverage, RunningAverageDict, compute_scale_and_shift, compute_errors, compute_metrics, ImageDataset

def main():
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser(description='Evaluate depth predictions')
    parser.add_argument('--path_video', type=str, required=True, help='Path to prediction video')
    # parser.add_argument('--path_pred', type=str, required=True, help='Path to prediction images')
    parser.add_argument('--gt_path', type=str, required=True, help='Path to ground truth video')
    parser.add_argument('--sport', type=str, required=False, default=None, help='Specify the sport you want to evaluate your predictions on (basket or foot)')
    args = parser.parse_args()

    path_pred = './temp/pred'
    gt_path = './temp/gt'
    V2I.ensure_path_exists(gt_path)
    V2I.ensure_path_exists(path_pred)
    if os.path.isdir(args.path_video):
        for pred_video in tqdm(os.listdir(args.path_video)):
             gt_video = pred_video.replace("color", "depth")
             pred_video_path = os.path.join(args.path_video, pred_video)
             gt_video_path = os.path.join(args.gt_path, gt_video)
             V2I.video_to_frames(pred_video_path, path_pred, frame_prefix=pred_video)
             V2I.video_to_frames(gt_video_path, gt_path, frame_prefix=pred_video)
    else:
        # 以下两行为测试代码（单个视频）
        V2I.video_to_frames(args.path_video, path_pred)
        V2I.video_to_frames(args.gt_path, gt_path)

    V2I.upsample_images_in_directory(path_pred)
    V2I.upsample_images_in_directory(gt_path)


    exclude_file = "basket_game_17_video_2_color_45.png"
    if args.sport is None:
        pred_files = [os.path.join(path_pred, f) for f in os.listdir(path_pred)
                      if os.path.isfile(os.path.join(path_pred, f)) and f != exclude_file]
        gt_files = [os.path.join(gt_path, f) for f in os.listdir(gt_path)
                    if os.path.isfile(os.path.join(gt_path, f))]
    else:
        pred_files = [os.path.join(path_pred, f) for f in os.listdir(path_pred)
                      if os.path.isfile(os.path.join(path_pred, f)) and args.sport in f and f != exclude_file]
        gt_files = [os.path.join(gt_path, f) for f in os.listdir(gt_path)
                    if os.path.isfile(os.path.join(gt_path, f)) and args.sport in f]
    pred_files.sort()
    gt_files.sort()


    print(f"Number of prediction files: {len(pred_files)}")
    print(f"Number of ground truth files: {len(gt_files)}")

    pred_dataset = ImageDataset(pred_files)
    gt_dataset = ImageDataset(gt_files)

    pred_loader = DataLoader(pred_dataset, batch_size=1, num_workers=2, pin_memory=True)
    gt_loader = DataLoader(gt_dataset, batch_size=1, num_workers=2, pin_memory=True)
    print("Loading done.")

    mask = torch.ones(1080, 1920, device=device)
    mask_score = False

    if args.sport == "basket":
        mask[870:1016, 1570:1829] = 0

    with open('test_score.txt', 'r') as f:
        file_contents = f.read().splitlines()

    mask.to(device).squeeze()

    print("Starting evaluation...")
    metrics = RunningAverageDict()
    i=0
    with torch.no_grad():
        for preds, gts in zip(pred_loader, gt_loader):
            mask_score = False
            preds, gts = preds.to(device).squeeze(), gts.to(device).squeeze()

            # Special case for some of the soccer test files that contain a score banner
            gt_file = gt_files[i]
            if gt_file in file_contents:
                mask[70:122, 95:612] = 0
                mask_score = True
            i+=1

            gts = gts / 255.
            if torch.all(gts == 1) or torch.all(gts == 0):
                continue
            preds = preds / 255.0

            scale, shift = compute_scale_and_shift(preds, gts, mask)
            scaled_predictions = scale.view(-1, 1, 1) * preds + shift.view(-1, 1, 1)

            metrics.update(compute_metrics(gts, scaled_predictions[0], mask_score, args.sport))

    print("Evaluation completed.")
    print("\n".join(f"{k}: {v}" for k, v in metrics.get_value().items()))

    V2I.delet_path('./temp')


if __name__ == "__main__":
    main()
