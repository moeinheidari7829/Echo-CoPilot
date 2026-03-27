from glob import glob
from tqdm import tqdm
import os
from os.path import join, basename
from collections import OrderedDict
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import argparse
import time

from PIL import Image
import SimpleITK as sitk
import torch
import torch.multiprocessing as mp
from sam2.build_sam import build_sam2_video_predictor_npz
from huggingface_hub import hf_hub_download

torch.set_float32_matmul_precision('high')
torch.manual_seed(2024)
torch.cuda.manual_seed(2024)
np.random.seed(2024)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--checkpoint',
    type=str,
    default="medsam2_recist", 
    help='medsam2 recist baseline checkpoint',
)
parser.add_argument(
    '--cfg',
    type=str,
    default="sam2/configs", 
    help='model config',
)
parser.add_argument(
    '-i',
    '--imgs_path',
    type=str,
    default="./data/validation_public_npz",
    help='imgs path',
)
parser.add_argument(
    '-o',
    '--pred_save_dir',
    type=str,
    default="./data/RECIST_pred",
    help='segs path',
)
parser.add_argument(
    '--shift',
    type=int,
    default=0,
)
parser.add_argument(
    '--sample_points',
    help='how to sample points for propagation',
    choices=['from_box', 'from_recist_n', 'from_recist_center', 'from_recist_3'],
    type=str,
    default="from_recist_n",
)
# add option to propagate with either box or point
parser.add_argument(
    '--propagate_with_box',
    default=False,
    action='store_true',
    help='whether to propagate with box'
)
parser.add_argument(
    '--save_nifti',
    default=False,
    action='store_true',
    help='whether to save nifti'
)
parser.add_argument(
    '--save_overlay',
    default=False,
    action='store_true',
    help='whether to save the overlay image'
)
parser.add_argument(
    '--num_workers',
    type=int,
    default=1,
)

args = parser.parse_args()
checkpoint = args.checkpoint
model_cfg = args.cfg
# make into absolute path
script_directory = os.path.dirname(os.path.abspath(__file__))
model_cfg = '//' + join(script_directory, model_cfg) # add additional one '/' in front of the path to avoid issues with relative paths
imgs_path = args.imgs_path
pred_save_dir = args.pred_save_dir
save_nifti = args.save_nifti
nifti_path = join(args.pred_save_dir, 'segs_nifti')
save_overlay = args.save_overlay
png_save_dir = join(args.pred_save_dir, 'png_overlay')
num_workers = args.num_workers
propagate_with_box = args.propagate_with_box
shift = args.shift
sample_points = args.sample_points

# download ckpt from huggingface
if checkpoint == 'medsam2_recist':
    ckpt_path = hf_hub_download(
        repo_id="wanglab/MedSAM2",
        filename="medsam2_FLARE25_RECIST_baseline.pt",
        cache_dir="./checkpoints"
    )
    model_cfg = model_cfg + "/sam2.1_hiera_t512.yaml"
    print(f'using medsam2 baseline recist model: {ckpt_path}')

predictor = build_sam2_video_predictor_npz(model_cfg, ckpt_path)

os.makedirs(pred_save_dir, exist_ok=True)
if save_overlay:
    os.makedirs(png_save_dir, exist_ok=True)
if save_nifti:
    os.makedirs(nifti_path, exist_ok=True)

def get_diameter_bbox(recist_per_lab, shift=0):
    H, W = recist_per_lab.shape
    # get line coordinates
    ys, xs = np.where(recist_per_lab > 0)
    coords = np.stack([xs, ys], axis=1)

    # Get endpoints
    p1 = coords[0]
    p2 = coords[-1]

    # Compute midpoint and line length
    center = ((p1 + p2) / 2).astype(int)
    diameter = np.linalg.norm(p1 - p2)
    half_side = int((diameter) / 2)

    # Get bounding box corners
    x_min = center[0] - half_side
    x_max = center[0] + half_side
    y_min = center[1] - half_side
    y_max = center[1] + half_side

    # clamp to image bounds
    x_min = max(0, x_min - shift)
    y_min = max(0, y_min - shift)
    x_max = min(W - 1, x_max + shift)
    y_max = min(H - 1, y_max + shift)

    return np.array([x_min, y_min, x_max, y_max])

def sample_points_in_bbox_grid(bbox: np.ndarray, n: int) -> np.ndarray:
    """
    Uniformly sample n grid-aligned (x, y) points inside the bbox.

    Args:
        bbox (np.ndarray): [x_min, y_min, x_max, y_max]
        n (int): Number of points to sample

    Returns:
        np.ndarray: shape (n, 2), each row is [x, y]
    """
    x_min, y_min, x_max, y_max = bbox
    grid_size = int(np.ceil(np.sqrt(n)))

    x_vals = np.linspace(x_min, x_max, grid_size, dtype=int)
    y_vals = np.linspace(y_min, y_max, grid_size, dtype=int)

    xv, yv = np.meshgrid(x_vals, y_vals)
    coords = np.stack([xv.ravel(), yv.ravel()], axis=1)

    return coords[:n]

def get_center_from_recist(recist_per_lab, scale=1):
    H, W = recist_per_lab.shape
    # get line coordinates
    ys, xs = np.where(recist_per_lab > 0)
    coords = np.stack([xs, ys], axis=1)

    # Get endpoints
    p1 = coords[0]
    p2 = coords[-1]

    # Compute midpoint and line length
    center = ((p1 + p2) / 2).astype(np.float32)
    return np.array([[center[0], center[1]]])

def get_n_points_from_recist(recist_per_lab, n=5):
    ys, xs = np.where(recist_per_lab > 0)
    coords = np.stack([xs, ys], axis=1)

    if len(coords) < n:
        raise ValueError(f"Cannot sample {n} points; RECIST line only has {len(coords)} pixels.")

    sampled_indices = np.random.choice(len(coords), size=n, replace=False)
    sampled_points = coords[sampled_indices]
    return sampled_points  # shape: (n, 2)

def get_center_and_endpoints_from_recist(recist_per_lab):
    ys, xs = np.where(recist_per_lab > 0)
    coords = np.stack([xs, ys], axis=1)

    if len(coords) < 2:
        raise ValueError("RECIST line must contain at least two points")

    # Endpoints
    p1 = coords[0].astype(np.float32)
    p2 = coords[-1].astype(np.float32)

    # Center (midpoint between endpoints)
    center = ((p1 + p2) / 2).astype(np.float32)

    return np.array([center, p1, p2])  # each is shape (2,)

def show_mask(mask, ax, mask_color=None, alpha=0.5):
    """
    show mask on the image

    Parameters
    ----------
    mask : numpy.ndarray
        mask of the image
    ax : matplotlib.axes.Axes
        axes to plot the mask
    mask_color : numpy.ndarray
        color of the mask
    alpha : float
        transparency of the mask
    """
    
    if mask_color is not None:
        color = np.concatenate([mask_color, np.array([alpha])], axis=0)
    else:
        color = np.array([251/255, 252/255, 30/255, alpha])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax, edgecolor='blue'):
    """
    show bounding box on the image

    Parameters
    ----------
    box : numpy.ndarray
        bounding box coordinates in the original image
    ax : matplotlib.axes.Axes
        axes to plot the bounding box
    edgecolor : str
        color of the bounding box
    """
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=edgecolor, facecolor=(0,0,0,0), lw=2))     

def resize_grayscale_to_rgb_and_resize(array, image_size):
    """
    Resize a 3D grayscale NumPy array to an RGB image and then resize it.
    
    Parameters:
        array (np.ndarray): Input array of shape (d, h, w).
        image_size (int): Desired size for the width and height.
    
    Returns:
        np.ndarray: Resized array of shape (d, 3, image_size, image_size).
    """
    d, h, w = array.shape
    resized_array = np.zeros((d, 3, image_size, image_size))
    
    for i in range(d):
        img_pil = Image.fromarray(array[i].astype(np.uint8))
        img_rgb = img_pil.convert("RGB")
        img_resized = img_rgb.resize((image_size, image_size))
        img_array = np.array(img_resized).transpose(2, 0, 1)  # (3, image_size, image_size)
        resized_array[i] = img_array
    
    return resized_array

@torch.inference_mode()
def infer_3d(img_npz_file):
    # start timer
    start_time = time.time()

    npz_name = basename(img_npz_file)
    print(f'processing {npz_name}')

    # load data from npz
    npz_data = np.load(img_npz_file, 'r', allow_pickle=True)
    print(f'available files: {npz_data.files}')
    spacing = npz_data['spacing']
    recist = npz_data['recist'] 
    img_3D_ori = npz_data['imgs']  # (D, H, W)
    assert np.max(img_3D_ori) < 256, f'input data should be in range [0, 255], but got {np.unique(img_3D_ori)}'

    segs_3D = np.zeros(img_3D_ori.shape, dtype=np.uint8)
    
    # get unique labels
    unique_labs = np.unique(recist)
    unique_labs = unique_labs[unique_labs != 0]
    print(f'unique_labs: {unique_labs}')

    # resize image to 512x512 and normalize
    video_height, video_width = img_3D_ori.shape[1:3]
    if video_height != 512 or video_width != 512:
        img_resized = resize_grayscale_to_rgb_and_resize(img_3D_ori, 512) #1024) #d, 3, 1024, 1024
    else:
        img_resized = img_3D_ori[:,None].repeat(3, axis=1) # d, 3, 1024, 1024
    img_resized = img_resized / 255.0
    img_resized = torch.from_numpy(img_resized).cuda()
    img_mean=(0.485, 0.456, 0.406)
    img_std=(0.229, 0.224, 0.225)
    img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None].cuda()
    img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None].cuda()
    img_resized -= img_mean
    img_resized /= img_std

    boxes_3D_ori = []
    all_points = []
    # iterate over each label
    for j, ulab in enumerate(unique_labs):
        recist_per_lab = (recist == ulab)*ulab
        if len(np.unique(recist_per_lab)) == 0:
            print(f'no recist for label {ulab} in {img_npz_file}, skipping...')
            continue

        z_mids = []
        idx = ulab

        z_indices = np.where(recist == ulab)[0]
        z_indices = np.unique(z_indices)
        assert len(z_indices) == 1, f'expected only one z index for recist=1, but got {z_indices}'
        z_mid = z_indices[0]
        z_min_per_lab = 0
        z_max_per_lab = img_3D_ori.shape[0] - 1
        z_mids.append(z_mid)
        
        # add prompt to initialize the predictor
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            inference_state = predictor.init_state(img_resized, video_height, video_width)
            if propagate_with_box:
                print('propagate with box')
                box_2d = get_diameter_bbox(recist_per_lab[z_mid], shift=shift)
                boxes_3D_ori.append([box_2d[0], box_2d[1], z_mid, box_2d[2], box_2d[3], z_mid])
                _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                                                    inference_state=inference_state,
                                                    frame_idx=z_mid,
                                                    obj_id=1,
                                                    box=box_2d,
                                                )
                mask_prompt = (out_mask_logits[0] > 0.0).squeeze(0).cpu().numpy().astype(np.uint8)
            else:
                print('propagate with point')
                if sample_points == 'from_box':
                    box_2d = get_diameter_bbox(recist_per_lab[z_mid], shift=shift)
                    boxes_3D_ori.append([box_2d[0], box_2d[1], z_mid, box_2d[2], box_2d[3], z_mid])
                    # sample points in the box
                    points = sample_points_in_bbox_grid(box_2d, n=9)
                elif sample_points == 'from_recist_n':
                    points = get_n_points_from_recist(recist_per_lab[z_mid], n=5)
                elif sample_points == 'from_recist_center':
                    points = get_center_from_recist(recist_per_lab[z_mid])
                elif sample_points == 'from_recist_3':
                    points = get_center_and_endpoints_from_recist(recist_per_lab[z_mid])
                else:
                    raise ValueError(f'unknown sample_points option: {sample_points}')
                
                labels = np.ones(len(points))   # all positive points
                all_points.append(points)
                _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                                                    inference_state=inference_state,
                                                    frame_idx=z_mid,
                                                    obj_id=1,
                                                    points=points,
                                                    labels=labels,
                                                )
                mask_prompt = (out_mask_logits[0] > 0.0).squeeze(0).cpu().numpy().astype(np.uint8)               

            # start inference propagation
            frame_idx, object_ids, masks = predictor.add_new_mask(inference_state, frame_idx=z_mid, obj_id=1, mask=mask_prompt)
            segs_3D[z_mid, ((masks[0] > 0.0).cpu().numpy())[0]] = idx
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, start_frame_idx=z_mid, reverse=False):
                segs_3D[(z_min_per_lab + out_frame_idx), (out_mask_logits[0] > 0.0).cpu().numpy()[0]] = idx
            # reverse process, delete old memory and initialize new predictor
            predictor.reset_state(inference_state)
            inference_state = predictor.init_state(img_resized, video_height, video_width)
            frame_idx, object_ids, masks = predictor.add_new_mask(inference_state, frame_idx=z_mid, obj_id=1, mask=mask_prompt)
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, start_frame_idx=z_mid, reverse=True):
                segs_3D[(z_min_per_lab + out_frame_idx), (out_mask_logits[0] > 0.0).cpu().numpy()[0]] = idx
            predictor.reset_state(inference_state)

    np.savez_compressed(join(pred_save_dir, npz_name), 
                        segs=segs_3D,
                        spacing=spacing,
                        )  
    # stop timer
    end_time = time.time()
    duration = end_time - start_time
    print(f'finished {npz_name} in {duration:.2f} seconds')

    if save_nifti:
        sitk_image = sitk.GetImageFromArray(img_3D_ori)
        # add spacing
        sitk_image.SetSpacing(spacing)
        sitk.WriteImage(sitk_image, os.path.join(nifti_path, npz_name.replace('.npz', '_imgs.nii.gz')))
        sitk_seg = sitk.GetImageFromArray(segs_3D)
        sitk_seg.SetSpacing(spacing)
        sitk.WriteImage(sitk_seg, os.path.join(nifti_path, npz_name.replace('.npz', '_segs.nii.gz')))

    if save_overlay:
        idx = random.sample(z_mids, 1)[0] 
        fig, ax = plt.subplots(1, 1, figsize=(5, 5)) 
        ax.imshow(img_3D_ori[idx], cmap='gray')
        ax.set_title("MedSAM2 Segmentation")
        ax.axis('off')

        for box_idx, label_id in enumerate(unique_labs):
            if np.sum(segs_3D[idx] == label_id) > 0:
                color = np.random.rand(3)
                if boxes_3D_ori:
                    x_min, y_min, z_min, x_max, y_max, z_max = boxes_3D_ori[box_idx]
                    box_viz = np.array([x_min, y_min, x_max, y_max])
                    if z_min <= idx <= z_max:
                        show_box(box_viz, ax, edgecolor=color)
                show_mask(segs_3D[idx] == label_id, ax, mask_color=color)
            else:
                print(f'no mask for file {img_npz_file=} {label_id=} {box_idx=}')

        # Add sampled points to ax
        if not propagate_with_box:
            last_pts = np.asarray(all_points[-1])
            if last_pts.size > 0:
                ax.scatter(
                    last_pts[:, 0], last_pts[:, 1],
                    s=20, c='red', marker='o',
                    edgecolors='white', linewidths=0.5
                )

        plt.tight_layout()
        plt.savefig(join(png_save_dir, npz_name.split(".")[0] + '_' + str(idx) + '.png'), dpi=300)
        plt.close()

    return npz_name, duration

if __name__ == '__main__':
    img_npz_files = sorted(glob(join(imgs_path, '*.npz'), recursive=True))
    img_npz_files = sorted(img_npz_files)
    print(f'number of files to process: {len(img_npz_files)}')

    output_dict = OrderedDict()
    output_dict['image'] = []
    output_dict['duration'] = []

    mp.set_start_method('spawn', force=True)
    with mp.Pool(num_workers) as pool:
        for npz_name, duration in tqdm(pool.imap(infer_3d, img_npz_files), total=len(img_npz_files)):
            output_dict['image'].append(npz_name)
            output_dict['duration'].append(duration)

    output_df = pd.DataFrame(output_dict)
    output_df.to_csv(join(pred_save_dir, 'inference_time.csv'), index=False)
    
