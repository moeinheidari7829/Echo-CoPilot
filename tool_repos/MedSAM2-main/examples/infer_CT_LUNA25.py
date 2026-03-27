"""
This script is used to run inference on the LUNA25 dataset using the MedSAM2 CT lesion model with point prompts.

Manually refined masks: https://huggingface.co/datasets/wanglab/LUNA25-MedSAM2
image: https://zenodo.org/records/14223624
annotation: https://zenodo.org/records/14673658
"""

from tqdm import tqdm
import os
from os.path import join
import pandas as pd
import numpy as np
import argparse

from PIL import Image
import SimpleITK as sitk
import torch
import torch.multiprocessing as mp
from sam2.build_sam import build_sam2_video_predictor_npz

torch.set_float32_matmul_precision('high')
torch.manual_seed(2024)
torch.cuda.manual_seed(2024)
np.random.seed(2024)

parser = argparse.ArgumentParser()

parser.add_argument(
    '--checkpoint',
    type=str,
    default="./hg_checkpoints/MedSAM2_CTLesion.pt", 
    help='checkpoint path',
)
parser.add_argument(
    '--cfg',
    type=str,
    default="configs/sam2.1/sam2.1_hiera_t512.yaml",
    help='model config',
)
parser.add_argument(
    '-i',
    '--imgs_path',
    type=str,
    default="/path/to/luna25_images",
    help='imgs path',
)
parser.add_argument(
    '-o',
    '--pred_save_dir',
    type=str,
    default="./segs/MedSAM2_release",
    help='segs path',
)
parser.add_argument(
    '--num_workers',
    type=int,
    default=2,
)
parser.add_argument(
    '--df_path',
    type=str,
    default='/path/to/LUNA25_Public_Training_Development_Data.csv',
)

args = parser.parse_args()
imsize = 512
df = pd.read_csv(args.df_path)
df = df[['SeriesInstanceUID', 'CoordX', 'CoordY', 'CoordZ']]

checkpoint = args.checkpoint
model_cfg = args.cfg
imgs_path = args.imgs_path
pred_save_dir = args.pred_save_dir
num_workers = args.num_workers
predictor = build_sam2_video_predictor_npz(model_cfg, checkpoint)
os.makedirs(pred_save_dir, exist_ok=True)


def preprocess(image_data, modality="CT", window_level=-750, window_width=1500):
    if modality == "CT":
        assert window_level is not None and window_width is not None, "CT modality requires window_level and window_width"
        lower_bound = window_level - window_width / 2
        upper_bound = window_level + window_width / 2
        image_data_pre = np.clip(image_data, lower_bound, upper_bound)
        image_data_pre = (
            (image_data_pre - np.min(image_data_pre))
            / (np.max(image_data_pre) - np.min(image_data_pre))
            * 255.0
        )
    else:
        lower_bound, upper_bound = np.percentile(
            image_data[image_data > 0], 0.5
        ), np.percentile(image_data[image_data > 0], 99.5)
        image_data_pre = np.clip(image_data, lower_bound, upper_bound)
        image_data_pre = (
            (image_data_pre - np.min(image_data_pre))
            / (np.max(image_data_pre) - np.min(image_data_pre))
            * 255.0
        )
        image_data_pre[image_data == 0] = 0
    
    return image_data_pre


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
def infer_3d(mha_name):
    print(f'processing {mha_name}')
    # get the corresponding keypoints of mha_name
    df_file = df[df['SeriesInstanceUID'] == mha_name.replace('.mha', '')]

    # read and preprocess the image
    sitk_img = sitk.ReadImage(join(imgs_path, mha_name))
    img_3D = preprocess(sitk.GetArrayFromImage(sitk_img))
    assert np.max(img_3D) < 256, f'input data should be in range [0, 255], but got {np.unique(img_3D)}'

    # initialize segmentation mask
    segs_3D = np.zeros(img_3D.shape, dtype=np.uint8)

    # resize and normalize the image
    video_height = img_3D.shape[1]
    video_width = img_3D.shape[2]
    if video_height != imsize or video_width != imsize:
        img_resized = resize_grayscale_to_rgb_and_resize(img_3D, imsize)  #d, 3, 512, 512
    else:
        img_resized = img_3D[:,None].repeat(3, axis=1) # d, 3, h, w
    img_resized = img_resized / 255.0
    img_resized = torch.from_numpy(img_resized).cuda()
    img_mean=(0.485, 0.456, 0.406)
    img_std=(0.229, 0.224, 0.225)
    img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None].cuda()
    img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None].cuda()
    img_resized -= img_mean
    img_resized /= img_std
    z_mids = []
    coords = []

    # for each point in the dataframe, get the corresponding 3D mask using keypoint prompts
    for index, (_, row) in enumerate(df_file.iterrows(), 1):
        
        x = row['CoordX']
        y = row['CoordY']
        z = row['CoordZ']
        # convert the coordinates to voxel coordinates
        voxel_x, voxel_y, voxel_z = sitk_img.TransformPhysicalPointToIndex((x, y, z))
        coords.append([voxel_x, voxel_y, voxel_z])
        z_mids.append(voxel_z)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            inference_state = predictor.init_state(img_resized, video_height, video_width)

            points = np.array([[voxel_x, voxel_y]], dtype=np.float32)
            # for labels, `1` means positive click and `0` means negative click
            labels = np.array([1], np.int32)
            # add point prompt
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=voxel_z,
                obj_id=1,
                points=points,
                labels=labels,
            )
            mask_prompt = (out_mask_logits[0] > 0.0).squeeze(0).cpu().numpy().astype(np.uint8)
                

            frame_idx, object_ids, masks = predictor.add_new_mask(inference_state, frame_idx=voxel_z, obj_id=1, mask=mask_prompt)
            segs_3D[voxel_z, ((masks[0] > 0.0).cpu().numpy())[0]] = index
            # propagate in the video
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, start_frame_idx=voxel_z, reverse=False):
                segs_3D[(out_frame_idx), (out_mask_logits[0] > 0.0).cpu().numpy()[0]] = index

            # reverse process, delete old memory and initialize new predictor
            predictor.reset_state(inference_state)
            inference_state = predictor.init_state(img_resized, video_height, video_width)
            frame_idx, object_ids, masks = predictor.add_new_mask(inference_state, frame_idx=voxel_z, obj_id=1, mask=mask_prompt)


            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, start_frame_idx=voxel_z, reverse=True):
                segs_3D[(out_frame_idx), (out_mask_logits[0] > 0.0).cpu().numpy()[0]] = index

            predictor.reset_state(inference_state)
    
    sitk.WriteImage(sitk.GetImageFromArray(segs_3D), join(pred_save_dir, mha_name.replace('.mha', '.nii.gz')))

    return



if __name__ == '__main__':
    img_mha_files = os.listdir(imgs_path)
    img_mha_files = [x for x in img_mha_files if x.endswith('.mha')]
    process_files = list(set(df['SeriesInstanceUID'].values))
    img_mha_files = [x for x in img_mha_files if x.replace('.mha', '') in process_files]

    print(f'number of files to process: {len(img_mha_files)}')

    mp.set_start_method('spawn', force=True)
    with mp.Pool(processes=num_workers) as pool:
        with tqdm(total=len(img_mha_files)) as pbar:
            for i, ret in tqdm(enumerate(pool.imap_unordered(infer_3d, img_mha_files))):
                pbar.update()



