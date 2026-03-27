"""
Gradio app for interactive medical video segmentation using MedSAM2.
Please use gradio==3.38.0
"""

import datetime
import gc
from glob import glob
import hashlib
import math
import multiprocessing as mp
import platform
import os
from os.path import basename, splitext, dirname
import threading
import time
os.environ["TORCH_CUDNN_SDPA_ENABLED"] = "1"
import shutil
import ffmpeg
from moviepy import ImageSequenceClip
import zipfile
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2_video_predictor
import cv2


user_processes = {}
PROCESS_TIMEOUT = datetime.timedelta(minutes=15)

def reset(seg_tracker):
    if seg_tracker is not None:
        predictor, inference_state, image_predictor = seg_tracker
        predictor.reset_state(inference_state)
        del predictor
        del inference_state
        del image_predictor
        del seg_tracker
        gc.collect()
        torch.cuda.empty_cache()
    return None, ({}, {}), None, None, 0, None, None, None, 0, 0, 

def extract_video_info(input_video):
    if input_video is None:
        return 4, 4, None, None, None, None, None
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, total_frames, None, None, None, None, None

def get_meta_from_video(session_id, input_video, scale_slider, config_path, checkpoint_path):
    output_dir = f'/tmp/output_frames/{session_id}'
    output_masks_dir = f'/tmp/output_masks/{session_id}'
    output_combined_dir = f'/tmp/output_combined/{session_id}'
    clear_folder(output_dir)
    clear_folder(output_masks_dir)
    clear_folder(output_combined_dir)
    if input_video is None:
        return None, ({}, {}), None, None, (4, 1, 4), None, None, None, 0, 0
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    frame_interval = max(1, int(fps // scale_slider))
    print(f"frame_interval: {frame_interval}")
    try:
        ffmpeg.input(input_video, hwaccel='cuda').output(
            os.path.join(output_dir, '%07d.jpg'), q=2, start_number=0, 
            vf=rf'select=not(mod(n\,{frame_interval}))', vsync='vfr'
        ).run()
    except:
        print(f"ffmpeg cuda err")
        ffmpeg.input(input_video).output(
            os.path.join(output_dir, '%07d.jpg'), q=2, start_number=0, 
            vf=rf'select=not(mod(n\,{frame_interval}))', vsync='vfr'
        ).run()

    first_frame_path = os.path.join(output_dir, '0000000.jpg')
    first_frame = cv2.imread(first_frame_path)
    first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)

    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
   
    predictor = build_sam2_video_predictor(config_path, checkpoint_path, device="cuda")
    sam2_model = build_sam2(config_path, checkpoint_path, device="cuda")
    image_predictor = SAM2ImagePredictor(sam2_model)
    inference_state = predictor.init_state(video_path=output_dir)
    predictor.reset_state(inference_state)
    return (predictor, inference_state, image_predictor), ({}, {}), first_frame_rgb, first_frame_rgb, (fps, frame_interval, total_frames), None, None, None, 0, 0

def mask2bbox(mask):
    if len(np.where(mask > 0)[0]) == 0:
        print(f'not mask')
        return np.array([0, 0, 0, 0]).astype(np.int64), False
    x_ = np.sum(mask, axis=0)
    y_ = np.sum(mask, axis=1)
    x0 = np.min(np.nonzero(x_)[0])
    x1 = np.max(np.nonzero(x_)[0])
    y0 = np.min(np.nonzero(y_)[0])
    y1 = np.max(np.nonzero(y_)[0])
    return np.array([x0, y0, x1, y1]).astype(np.int64), True

def sam_stroke(session_id, seg_tracker, drawing_board, last_draw, frame_num, ann_obj_id):
    predictor, inference_state, image_predictor = seg_tracker
    image_path = f'/tmp/output_frames/{session_id}/{frame_num:07d}.jpg'
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    display_image = drawing_board["image"]
    image_predictor.set_image(image)
    input_mask = drawing_board["mask"]
    input_mask[input_mask != 0] = 255
    if last_draw is not None:
        diff_mask = cv2.absdiff(input_mask, last_draw)
        input_mask = diff_mask
    bbox, hasMask = mask2bbox(input_mask[:, :, 0]) 
    if not hasMask :
        return seg_tracker, display_image, display_image, None
    masks, scores, logits = image_predictor.predict( point_coords=None, point_labels=None, box=bbox[None, :], multimask_output=False,)
    mask = masks > 0.0
    masked_frame = show_mask(mask, display_image, ann_obj_id)
    masked_with_rect = draw_rect(masked_frame, bbox, ann_obj_id)
    frame_idx, object_ids, masks = predictor.add_new_mask(inference_state, frame_idx=frame_num, obj_id=ann_obj_id, mask=mask[0])
    last_draw = drawing_board["mask"]
    return seg_tracker, masked_with_rect, masked_with_rect, last_draw

def draw_rect(image, bbox, obj_id):
    cmap = plt.get_cmap("tab10")
    color = np.array(cmap(obj_id)[:3])
    rgb_color = tuple(map(int, (color[:3] * 255).astype(np.uint8)))
    inv_color = tuple(map(int, (255 - color[:3] * 255).astype(np.uint8)))
    x0, y0, x1, y1 = bbox
    image_with_rect = cv2.rectangle(image.copy(), (x0, y0), (x1, y1), rgb_color, thickness=2)
    return image_with_rect

def sam_click(session_id, seg_tracker, frame_num, point_mode, click_stack, ann_obj_id, point):
    points_dict, labels_dict = click_stack
    predictor, inference_state, image_predictor = seg_tracker
    ann_frame_idx = frame_num  # the frame index we interact with
    print(f'ann_frame_idx: {ann_frame_idx}')
    if point_mode == "Positive":
        label = np.array([1], np.int32)
    else:
        label = np.array([0], np.int32)

    if ann_frame_idx not in points_dict:
        points_dict[ann_frame_idx] = {}
    if ann_frame_idx not in labels_dict:
        labels_dict[ann_frame_idx] = {}

    if ann_obj_id not in points_dict[ann_frame_idx]:
        points_dict[ann_frame_idx][ann_obj_id] = np.empty((0, 2), dtype=np.float32)
    if ann_obj_id not in labels_dict[ann_frame_idx]:
        labels_dict[ann_frame_idx][ann_obj_id] = np.empty((0,), dtype=np.int32)

    points_dict[ann_frame_idx][ann_obj_id] = np.append(points_dict[ann_frame_idx][ann_obj_id], point, axis=0)
    labels_dict[ann_frame_idx][ann_obj_id] = np.append(labels_dict[ann_frame_idx][ann_obj_id], label, axis=0)

    click_stack = (points_dict, labels_dict)

    frame_idx, out_obj_ids, out_mask_logits = predictor.add_new_points(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points_dict[ann_frame_idx][ann_obj_id],
        labels=labels_dict[ann_frame_idx][ann_obj_id],
    )

    image_path = f'/tmp/output_frames/{session_id}/{ann_frame_idx:07d}.jpg'
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    masked_frame = image.copy()
    for i, obj_id in enumerate(out_obj_ids):
        mask = (out_mask_logits[i] > 0.0).cpu().numpy()
        masked_frame = show_mask(mask, image=masked_frame, obj_id=obj_id)
    masked_frame_with_markers = draw_markers(masked_frame, points_dict[ann_frame_idx], labels_dict[ann_frame_idx])

    return seg_tracker, masked_frame_with_markers, masked_frame_with_markers, click_stack

def draw_markers(image, points_dict, labels_dict):
    cmap = plt.get_cmap("tab10")
    image_h, image_w = image.shape[:2]
    marker_size = max(1, int(min(image_h, image_w) * 0.05))

    for obj_id in points_dict:
        color = np.array(cmap(obj_id)[:3])
        rgb_color = tuple(map(int, (color[:3] * 255).astype(np.uint8)))
        inv_color = tuple(map(int, (255 - color[:3] * 255).astype(np.uint8)))
        for point, label in zip(points_dict[obj_id], labels_dict[obj_id]):
            x, y = int(point[0]), int(point[1])
            if label == 1:
                cv2.drawMarker(image, (x, y), inv_color, markerType=cv2.MARKER_CROSS, markerSize=marker_size, thickness=2)
            else:
                cv2.drawMarker(image, (x, y), inv_color, markerType=cv2.MARKER_TILTED_CROSS, markerSize=int(marker_size / np.sqrt(2)), thickness=2)
    
    return image

def show_mask(mask, image=None, obj_id=None):
    cmap = plt.get_cmap("tab10")
    cmap_idx = 0 if obj_id is None else obj_id
    color = np.array([*cmap(cmap_idx)[:3], 0.6])
    
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    mask_image = (mask_image * 255).astype(np.uint8)
    if image is not None:
        image_h, image_w = image.shape[:2]
        if (image_h, image_w) != (h, w):
            raise ValueError(f"Image dimensions ({image_h}, {image_w}) and mask dimensions ({h}, {w}) do not match")
        colored_mask = np.zeros_like(image, dtype=np.uint8)
        for c in range(3):
            colored_mask[..., c] = mask_image[..., c]
        alpha_mask = mask_image[..., 3] / 255.0
        for c in range(3):
            image[..., c] = np.where(alpha_mask > 0, (1 - alpha_mask) * image[..., c] + alpha_mask * colored_mask[..., c], image[..., c])
        return image
    return mask_image

def show_res_by_slider(session_id, frame_per, click_stack):
    image_path = f'/tmp/output_frames/{session_id}'
    output_combined_dir = f'/tmp/output_combined/{session_id}'
    
    combined_frames = sorted([os.path.join(output_combined_dir, img_name) for img_name in os.listdir(output_combined_dir)])
    if combined_frames:
        output_masked_frame_path = combined_frames
    else:
        original_frames = sorted([os.path.join(image_path, img_name) for img_name in os.listdir(image_path)])
        output_masked_frame_path = original_frames
       
    total_frames_num = len(output_masked_frame_path)
    if total_frames_num == 0:
        print("No output results found")
        return None, None, 0
    else:
        frame_num = math.floor(total_frames_num * frame_per)
        if frame_num >= total_frames_num:
            frame_num = total_frames_num - 1
        chosen_frame_path = output_masked_frame_path[frame_num]
        print(f"{chosen_frame_path}")
        chosen_frame_show = cv2.imread(chosen_frame_path)
        chosen_frame_show = cv2.cvtColor(chosen_frame_show, cv2.COLOR_BGR2RGB)
        points_dict, labels_dict = click_stack
        if frame_num in points_dict and frame_num in labels_dict:
            chosen_frame_show = draw_markers(chosen_frame_show, points_dict[frame_num], labels_dict[frame_num])
        return chosen_frame_show, chosen_frame_show, frame_num

def clear_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)

def zip_folder(folder_path, output_zip_path):
    with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_STORED) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, folder_path))

def tracking_objects(session_id, seg_tracker, frame_num, input_video):
    output_dir = f'/tmp/output_frames/{session_id}'
    output_masks_dir = f'/tmp/output_masks/{session_id}'
    output_combined_dir = f'/tmp/output_combined/{session_id}'
    output_files_dir = f'/tmp/output_files/{session_id}'
    output_video_path = f'{output_files_dir}/output_video.mp4'
    output_zip_path = f'{output_files_dir}/output_masks.zip'
    clear_folder(output_masks_dir)
    clear_folder(output_combined_dir)
    clear_folder(output_files_dir)
    video_segments = {}
    predictor, inference_state, image_predictor = seg_tracker
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    frame_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.jpg')])
    # for frame_idx in sorted(video_segments.keys()):
    for frame_file in frame_files:
        frame_idx = int(os.path.splitext(frame_file)[0])
        frame_path = os.path.join(output_dir, frame_file)
        image = cv2.imread(frame_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masked_frame = image.copy()
        if frame_idx in video_segments:
            for obj_id, mask in video_segments[frame_idx].items():
                masked_frame = show_mask(mask, image=masked_frame, obj_id=obj_id)
                mask_output_path = os.path.join(output_masks_dir, f'{obj_id}_{frame_idx:07d}.png')
                cv2.imwrite(mask_output_path, show_mask(mask))
        combined_output_path = os.path.join(output_combined_dir, f'{frame_idx:07d}.png')
        combined_image_bgr = cv2.cvtColor(masked_frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(combined_output_path, combined_image_bgr)
        if frame_idx == frame_num:
            final_masked_frame = masked_frame

    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    # output_frames = int(total_frames * scale_slider)
    output_frames = len([name for name in os.listdir(output_combined_dir) if os.path.isfile(os.path.join(output_combined_dir, name)) and name.endswith('.png')])
    out_fps = fps * output_frames / total_frames

    # ffmpeg.input(os.path.join(output_combined_dir, '%07d.png'), framerate=out_fps).output(output_video_path, vcodec='h264_nvenc', pix_fmt='yuv420p').run()

    # fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # out = cv2.VideoWriter(output_video_path, fourcc, out_fps, (frame_width, frame_height))
    # for i in range(output_frames):
    #     frame_path = os.path.join(output_combined_dir, f'{i:07d}.png')
    #     frame = cv2.imread(frame_path)
    #     out.write(frame)
    # out.release()

    image_files = [os.path.join(output_combined_dir, f'{i:07d}.png') for i in range(output_frames)]
    clip = ImageSequenceClip(image_files, fps=out_fps)
    clip.write_videofile(output_video_path, codec="libx264", fps=out_fps)

    zip_folder(output_masks_dir, output_zip_path)
    print("done")
    return final_masked_frame, final_masked_frame, output_video_path, output_video_path, output_zip_path, ({}, {})

def increment_ann_obj_id(max_obj_id):
    max_obj_id += 1
    ann_obj_id = max_obj_id
    return ann_obj_id, max_obj_id

def update_current_id(ann_obj_id):
    return ann_obj_id

def drawing_board_get_input_first_frame(input_first_frame):
    return input_first_frame

def process_video(queue, result_queue, session_id):
    seg_tracker = None
    click_stack = ({}, {})
    frame_num = int(0)
    ann_obj_id = int(0)
    last_draw = None
    while True:
        task = queue.get() 
        if task["command"] == "exit":
            print(f"Process for {session_id} exiting.")
            break
        elif task["command"] == "extract_video_info":
            input_video = task["input_video"]
            fps, total_frames, input_first_frame, drawing_board, output_video, output_mp4, output_mask = extract_video_info(input_video)
            result_queue.put({"fps": fps, "total_frames": total_frames, "input_first_frame": input_first_frame, "drawing_board": drawing_board, "output_video": output_video, "output_mp4": output_mp4, "output_mask": output_mask})
        elif task["command"] == "get_meta_from_video":
            input_video = task["input_video"]
            scale_slider = task["scale_slider"]
            config_path = task["config_path"]
            checkpoint_path = task["checkpoint_path"]
            seg_tracker, click_stack, input_first_frame, drawing_board, frame_per, output_video, output_mp4, output_mask, ann_obj_id, max_obj_id = get_meta_from_video(session_id, input_video, scale_slider, config_path, checkpoint_path)
            result_queue.put({"input_first_frame": input_first_frame, "drawing_board": drawing_board, "frame_per": frame_per, "output_video": output_video, "output_mp4": output_mp4, "output_mask": output_mask, "ann_obj_id": ann_obj_id, "max_obj_id": max_obj_id})
        elif task["command"] == "sam_stroke":
            drawing_board = task["drawing_board"]
            last_draw = task["last_draw"]
            frame_num = task["frame_num"]
            ann_obj_id = task["ann_obj_id"]
            seg_tracker, input_first_frame, drawing_board, last_draw = sam_stroke(session_id, seg_tracker, drawing_board, last_draw, frame_num, ann_obj_id)
            result_queue.put({"input_first_frame": input_first_frame, "drawing_board": drawing_board, "last_draw": last_draw})
        elif task["command"] == "sam_click":
            frame_num = task["frame_num"]
            point_mode = task["point_mode"]
            click_stack = task["click_stack"]
            ann_obj_id = task["ann_obj_id"]
            point = task["point"]
            seg_tracker, input_first_frame, drawing_board, last_draw = sam_click(session_id, seg_tracker, frame_num, point_mode, click_stack, ann_obj_id, point)
            result_queue.put({"input_first_frame": input_first_frame, "drawing_board": drawing_board, "last_draw": last_draw})
        elif task["command"] == "increment_ann_obj_id":
            max_obj_id = task["max_obj_id"]
            ann_obj_id, max_obj_id = increment_ann_obj_id(max_obj_id)
            result_queue.put({"ann_obj_id": ann_obj_id, "max_obj_id": max_obj_id})
        elif task["command"] == "update_current_id":
            ann_obj_id = task["ann_obj_id"]
            ann_obj_id = update_current_id(ann_obj_id)
            result_queue.put({"ann_obj_id": ann_obj_id})
        elif task["command"] == "drawing_board_get_input_first_frame":
            input_first_frame = task["input_first_frame"]
            input_first_frame = drawing_board_get_input_first_frame(input_first_frame)
            result_queue.put({"input_first_frame": input_first_frame})
        elif task["command"] == "reset":
            seg_tracker, click_stack, input_first_frame, drawing_board, frame_per, output_video, output_mp4, output_mask, ann_obj_id, max_obj_id = reset(seg_tracker)
            result_queue.put({"click_stack": click_stack, "input_first_frame": input_first_frame, "drawing_board": drawing_board, "frame_per": frame_per, "output_video": output_video, "output_mp4": output_mp4, "output_mask": output_mask, "ann_obj_id": ann_obj_id, "max_obj_id": max_obj_id})
        elif task["command"] == "show_res_by_slider":
            frame_per = task["frame_per"]
            click_stack = task["click_stack"]
            input_first_frame, drawing_board, frame_num = show_res_by_slider(session_id, frame_per, click_stack)
            result_queue.put({"input_first_frame": input_first_frame, "drawing_board": drawing_board, "frame_num": frame_num})
        elif task["command"] == "tracking_objects":
            frame_num = task["frame_num"]
            input_video = task["input_video"]
            input_first_frame, drawing_board, output_video, output_mp4, output_mask, click_stack = tracking_objects(session_id, seg_tracker, frame_num, input_video)
            result_queue.put({"input_first_frame": input_first_frame, "drawing_board": drawing_board, "output_video": output_video, "output_mp4": output_mp4, "output_mask": output_mask, "click_stack": click_stack})
        else:
            print(f"Unknown command {task['command']} for {session_id}")
            result_queue.put("Unknown command")

def start_process(session_id):
    if session_id not in user_processes:
        queue = mp.Queue()
        result_queue = mp.Queue()
        process = mp.Process(target=process_video, args=(queue, result_queue, session_id))
        process.start()
        user_processes[session_id] = {
            "process": process,
            "queue": queue,
            "result_queue": result_queue,
            "last_active": datetime.datetime.now()
        }
    else:
        user_processes[session_id]["last_active"] = datetime.datetime.now()
    return user_processes[session_id]["queue"]

# def clean_up_processes(session_id, init_clean = False):
#     now = datetime.datetime.now()
#     to_remove = []
#     for s_id, process_info in user_processes.items():
#         if (now - process_info["last_active"] > PROCESS_TIMEOUT) or (s_id == session_id and init_clean):
#             process_info["queue"].put({"command": "exit"})
#             process_info["process"].terminate()
#             process_info["process"].join()
#             to_remove.append(s_id)
#     for s_id in to_remove:
#         del user_processes[s_id]
#         print(f"Cleaned up process for session {s_id}.")
        
def monitor_and_cleanup_processes():
    while True:
        now = datetime.datetime.now()
        to_remove = []
        for session_id, process_info in user_processes.items():
            if now - process_info["last_active"] > PROCESS_TIMEOUT:
                process_info["queue"].put({"command": "exit"})
                process_info["process"].terminate()
                process_info["process"].join()
                to_remove.append(session_id)
        for session_id in to_remove:
            del user_processes[session_id]
            print(f"Automatically cleaned up process for session {session_id}.")
        time.sleep(10)

def seg_track_app():
    # Only supports gradio==3.38.0
    import gradio as gr
    
    def extract_session_id_from_request(request: gr.Request):
        session_id = hashlib.sha256(f'{request.client.host}:{request.client.port}'.encode('utf-8')).hexdigest()
        # cookies = request.kwargs["headers"].get('cookie', '')
        # session_id = None
        # if '_gid=' in cookies:
        #     session_id = cookies.split('_gid=')[1].split(';')[0]
        # else:
        #     session_id = str(uuid.uuid4())
        print(f"session_id {session_id}")
        return session_id

    def handle_extract_video_info(session_id, input_video):
        # clean_up_processes(session_id, init_clean=True)
        if input_video == None:
            return 0, 0, {
            "minimum": 0.0,
            "maximum": 100,
            "step": 0.01,
            "value": 0.0,
        }, None, None, None, None, None
        queue = start_process(session_id)
        result_queue = user_processes[session_id]["result_queue"]
        queue.put({"command": "extract_video_info", "input_video": input_video})
        result = result_queue.get()
        fps = result.get("fps")
        total_frames = result.get("total_frames")
        input_first_frame = result.get("input_first_frame")
        drawing_board = result.get("drawing_board")
        output_video = result.get("output_video")
        output_mp4 = result.get("output_mp4")
        output_mask = result.get("output_mask")
        scale_slider = gr.Slider.update(minimum=1.0,
                                    maximum=fps,
                                    step=1.0,
                                    value=fps,)
        frame_per = gr.Slider.update(minimum= 0.0,
                                maximum= total_frames / fps,
                                step=1.0/fps,
                                value=0.0,)
        slider_state = {
            "minimum": 0.0,
            "maximum": total_frames / fps,
            "step": 1.0/fps,
            "value": 0.0,
        }
        return scale_slider, frame_per, slider_state, input_first_frame, drawing_board, output_video, output_mp4, output_mask

    def handle_get_meta_from_video(session_id, input_video, scale_slider, selected_config, selected_checkpoint):
        config_path = config_file_map[selected_config]
        checkpoint_path = checkpoint_file_map[selected_checkpoint]
        # clean_up_processes(session_id)
        queue = start_process(session_id)
        result_queue = user_processes[session_id]["result_queue"]
        queue.put({"command": "get_meta_from_video", "input_video": input_video, "scale_slider": scale_slider, "config_path": config_path, "checkpoint_path": checkpoint_path})
        result = result_queue.get()
        input_first_frame = result.get("input_first_frame")
        drawing_board = result.get("drawing_board")
        (fps, frame_interval, total_frames) = result.get("frame_per")
        output_video = result.get("output_video")
        output_mp4 = result.get("output_mp4")
        output_mask = result.get("output_mask")
        ann_obj_id = result.get("ann_obj_id")
        max_obj_id = result.get("max_obj_id")
        frame_per = gr.Slider.update(minimum= 0.0,
                                maximum= total_frames / fps,
                                step=frame_interval / fps / 2,
                                value=0.0,)
        slider_state = {
            "minimum": 0.0,
            "maximum": total_frames / fps,
            "step": frame_interval/fps / 2 ,
            "value": 0.0,
        }
        obj_id_slider = gr.Slider.update(
                                    maximum=max_obj_id, 
                                    value=ann_obj_id
                                )
        return input_first_frame, drawing_board, frame_per, slider_state, output_video, output_mp4, output_mask, ann_obj_id, max_obj_id, obj_id_slider

    def handle_sam_stroke(session_id, drawing_board, last_draw, frame_num, ann_obj_id):
        # clean_up_processes(session_id)
        queue = start_process(session_id)
        result_queue = user_processes[session_id]["result_queue"]
        queue.put({"command": "sam_stroke", "drawing_board": drawing_board, "last_draw": last_draw, "frame_num": frame_num, "ann_obj_id": ann_obj_id})
        result = result_queue.get()
        input_first_frame = result.get("input_first_frame")
        drawing_board = result.get("drawing_board")
        last_draw = result.get("last_draw")
        return input_first_frame, drawing_board, last_draw

    def handle_sam_click(session_id, frame_num, point_mode, click_stack, ann_obj_id, evt: gr.SelectData):
        # clean_up_processes(session_id)
        queue = start_process(session_id)
        result_queue = user_processes[session_id]["result_queue"]
        point = np.array([[evt.index[0], evt.index[1]]], dtype=np.float32)
        queue.put({"command": "sam_click", "frame_num": frame_num, "point_mode": point_mode, "click_stack": click_stack, "ann_obj_id": ann_obj_id, "point": point})
        result = result_queue.get()
        input_first_frame = result.get("input_first_frame")
        drawing_board = result.get("drawing_board")
        last_draw = result.get("last_draw")
        return input_first_frame, drawing_board, last_draw

    def handle_increment_ann_obj_id(session_id, max_obj_id):
        # clean_up_processes(session_id)
        queue = start_process(session_id)
        result_queue = user_processes[session_id]["result_queue"]
        queue.put({"command": "increment_ann_obj_id", "max_obj_id": max_obj_id})
        result = result_queue.get()
        ann_obj_id = result.get("ann_obj_id")
        max_obj_id = result.get("max_obj_id")
        obj_id_slider = gr.Slider.update(maximum=max_obj_id, value=ann_obj_id)
        return ann_obj_id, max_obj_id, obj_id_slider

    def handle_update_current_id(session_id, ann_obj_id):
        # clean_up_processes(session_id)
        queue = start_process(session_id)
        result_queue = user_processes[session_id]["result_queue"]
        queue.put({"command": "update_current_id", "ann_obj_id": ann_obj_id})
        result = result_queue.get()
        ann_obj_id = result.get("ann_obj_id")
        return ann_obj_id

    def handle_drawing_board_get_input_first_frame(session_id, input_first_frame):
        # clean_up_processes(session_id)
        queue = start_process(session_id)
        result_queue = user_processes[session_id]["result_queue"]
        queue.put({"command": "drawing_board_get_input_first_frame", "input_first_frame": input_first_frame})
        result = result_queue.get()
        input_first_frame = result.get("input_first_frame")
        return input_first_frame

    def handle_reset(session_id):
        # clean_up_processes(session_id)
        queue = start_process(session_id)
        result_queue = user_processes[session_id]["result_queue"]
        queue.put({"command": "reset"})
        result = result_queue.get()
        click_stack = result.get("click_stack")
        input_first_frame = result.get("input_first_frame")
        drawing_board = result.get("drawing_board")
        slider_state = {
            "minimum": 0.0,
            "maximum": 100,
            "step": 0.01,
            "value": 0.0,
        }
        output_video = result.get("output_video")
        output_mp4 = result.get("output_mp4")
        output_mask = result.get("output_mask")
        ann_obj_id = result.get("ann_obj_id")
        max_obj_id = result.get("max_obj_id")
        obj_id_slider = gr.Slider.update(
                            maximum=max_obj_id, 
                            value=ann_obj_id)
        return click_stack, input_first_frame, drawing_board, frame_per, slider_state, output_video, output_mp4, output_mask, ann_obj_id, max_obj_id, obj_id_slider

    def handle_show_res_by_slider(session_id, frame_per, slider_state, click_stack):
        # clean_up_processes(session_id)
        queue = start_process(session_id)
        frame_per = frame_per/slider_state["maximum"]
        result_queue = user_processes[session_id]["result_queue"]
        queue.put({"command": "show_res_by_slider", "frame_per": frame_per, "click_stack": click_stack})
        result = result_queue.get()
        input_first_frame = result.get("input_first_frame")
        drawing_board = result.get("drawing_board")
        frame_num = result.get("frame_num")
        return input_first_frame, drawing_board, frame_num

    def handle_tracking_objects(session_id, frame_num, input_video):
        # clean_up_processes(session_id)
        queue = start_process(session_id)
        result_queue = user_processes[session_id]["result_queue"]
        queue.put({"command": "tracking_objects", "frame_num": frame_num, "input_video": input_video})
        result = result_queue.get()
        input_first_frame = result.get("input_first_frame")
        drawing_board = result.get("drawing_board")
        output_video = result.get("output_video")
        output_mp4 = result.get("output_mp4")
        output_mask = result.get("output_mask")
        click_stack = result.get("click_stack")
        return input_first_frame, drawing_board, output_video, output_mp4, output_mask, click_stack

    ##########################################################
    ######################  Front-end ########################
    ##########################################################
    css = """
    #input_output_video video {
        max-height: 550px;
        max-width: 100%;
        height: auto;
    }
    """

    if platform.system() == "Windows":
        config_path = os.path.abspath(os.environ.get("CONFIG_PATH", "sam2/configs/"))
        checkpoint_path = os.environ.get("CHECKPOINT_PATH", "checkpoints/")

        config_files = glob(os.path.join(config_path, "*.yaml"))
        config_files.sort(key=lambda x: '_t.' not in basename(x))

        checkpoint_files = glob(os.path.join(checkpoint_path, "*.pt"))
        checkpoint_files.sort(key=lambda x: 'tiny' not in basename(x))

        medsam_checkpoints = glob("checkpoints/*.pt")
    else:
        config_path = "/" + os.path.abspath(os.environ.get("CONFIG_PATH", "./sam2/configs/"))
        checkpoint_path = os.environ.get("CHECKPOINT_PATH", "./checkpoints")

        config_files = glob(os.path.join(config_path, "*.yaml"))
        config_files.sort(key=lambda x: '_t.' not in basename(x))

        checkpoint_files = glob(os.path.join(checkpoint_path, "*.pt"))
        checkpoint_files.sort(key=lambda x: 'tiny' not in basename(x))

        medsam_checkpoints = glob("./checkpoints/*.pt")

    config_display = [splitext(basename(f))[0] for f in config_files]
    medsam_display = [
        f"{os.path.basename(dirname(dirname(path)))} / {splitext(basename(path))[0]}"
        for path in medsam_checkpoints
    ]
    checkpoint_display = [
        splitext(basename(f))[0] for f in checkpoint_files
    ] + medsam_display
    checkpoint_files.extend(medsam_checkpoints)

    config_file_map = dict(zip(config_display, config_files))
    checkpoint_file_map = dict(zip(checkpoint_display, checkpoint_files))

    app = gr.Blocks(css=css)
    with app:
        session_id = gr.State()
        app.load(extract_session_id_from_request, None, session_id)
        gr.Markdown(
            '''
            <div style="text-align:center; margin-bottom:20px;">
                <span style="font-size:3em; font-weight:bold;">MedSAM2: Segment Anything in 3D Medical Images and Videos</span>
            </div>
            <div style="text-align:center; margin-bottom:20px;">
                <a href="https://github.com/bowang-lab/MedSAM/tree/MedSAM2">
                    <img src="https://badges.aleen42.com/src/github.svg" alt="GitHub" style="display:inline-block; margin-right:10px;">
                </a>
                <a href="https://arxiv.org/abs/2408.03322">
                    <img src="https://img.shields.io/badge/arXiv-2408.03322-green?style=plastic" alt="Paper" style="display:inline-block; margin-right:10px;">
                </a>
                <a href="https://github.com/bowang-lab/MedSAMSlicer/tree/MedSAM2">
                    <img src="https://img.shields.io/badge/3D-Slicer-Plugin" alt="3D Slicer Plugin" style="display:inline-block; margin-right:10px;">
                </a>
            </div>
            <div style="text-align:left; margin-bottom:20px;">
                This API supports using box (generated by scribble) and point prompts for medical video segmentation.
            </div>
            <div style="margin-bottom:20px;">
                <ol style="list-style:none; padding-left:0;">
                    <li>1. Upload video file</li>
                    <li>2. Select model size and downsample frame rate and run <b>Preprocess</b></li>
                    <li>3. Use <b>Stroke to Box Prompt</b> to draw box on the first frame or <b>Point Prompt</b> to click on the first frame.</li>
                    <li>&nbsp;&nbsp;&nbsp;Note: The bounding rectangle of the stroke should be able to cover the segmentation target.</li>
                    <li>4. Click <b>Segment</b> to get the segmentation result</li>
                    <li>5. Click <b>Add New Object</b> to add new object</li>
                    <li>6. Click <b>Start Tracking</b> to track objects in the video</li>
                    <li>7. Click <b>Reset</b> to reset the app</li>
                    <li>8. Download the video with segmentation results</li>
                </ol>
            </div>
            <div style="text-align:left; line-height:1.8;">
                If you find these tools useful, please consider citing the following papers:
            </div>
            <div style="text-align:left; line-height:1.8;">
                Ravi, N., Gabeur, V., Hu, Y.T., Hu, R., Ryali, C., Ma, T., Khedr, H., Rädle, R., Rolland, C., Gustafson, L., Mintun, E., Pan, J., Alwala, K.V., Carion, N., Wu, C.Y., Girshick, R., Dollár, P., Feichtenhofer, C.: SAM 2: Segment Anything in Images and Videos. ICLR 2025
            </div>            
            <div style="text-align:left; line-height:1.8;">
                Ma, J.*, Yang, Z.*, Kim, S., Chen, B., Baharoon, M., Fallahpour, A, Asakereh, R., Lyu, H., Wang, B.: MedSAM2: Segment Anything in Medical Images and Videos. arXiv preprint (2025)
            </div> 
            '''
        )

        click_stack = gr.State(({}, {}))
        frame_num = gr.State(value=(int(0)))
        ann_obj_id = gr.State(value=(int(0)))
        max_obj_id = gr.State(value=(int(0)))
        last_draw = gr.State(None)
        slider_state = gr.State(value={
            "minimum": 0.0,
            "maximum": 100,
            "step": 0.01,
            "value": 0.0,
        })

        with gr.Row():
            with gr.Column(scale=0.5):
                with gr.Row():
                    tab_video_input = gr.Tab(label="Video input")
                    with tab_video_input:
                        input_video = gr.Video(label='Input video', type=["mp4", "mov", "avi"], elem_id="input_output_video")
                        with gr.Row():
                            # checkpoint = gr.Dropdown(label="Model Size", choices=["tiny", "small", "base-plus", "large"], value="tiny")
                            config_dropdown = gr.Dropdown(
                                choices=config_display, 
                                value=config_display[0],
                                label="Select Config File"
                            )

                            checkpoint_dropdown = gr.Dropdown(
                                choices=checkpoint_display, 
                                value=checkpoint_display[0],
                                label="Select Checkpoint File"
                            )
                            scale_slider = gr.Slider(
                                label="Downsampe Frame Rate (fps)",
                                minimum=0.0,
                                maximum=1.0,
                                step=0.25,
                                value=1.0,
                                interactive=True
                            )
                            preprocess_button = gr.Button(
                                value="Preprocess",
                                interactive=True,
                            )

                with gr.Row():
                    tab_stroke = gr.Tab(label="Stroke to Box Prompt")
                    with tab_stroke:
                        drawing_board = gr.Image(label='Drawing Board', tool="sketch", brush_radius=10, interactive=True)
                        with gr.Row():
                            seg_acc_stroke = gr.Button(value="Segment", interactive=True)
                            
                    tab_click = gr.Tab(label="Point Prompt")
                    with tab_click:
                        input_first_frame = gr.Image(label='Segment result of first frame',interactive=True).style(height=550)
                        with gr.Row():
                            point_mode = gr.Radio(
                                        choices=["Positive",  "Negative"],
                                        value="Positive",
                                        label="Point Prompt",
                                        interactive=True)
                            
                with gr.Row():
                    with gr.Column():
                        frame_per = gr.Slider(
                            label = "Time (seconds)",
                            minimum= 0.0,
                            maximum= 100.0,
                            step=0.01,
                            value=0.0,
                        )
                        with gr.Row():
                            with gr.Column():
                                obj_id_slider = gr.Slider(
                                    minimum=0, 
                                    maximum=0, 
                                    step=1, 
                                    interactive=True,
                                    label="Current Object ID"
                                )
                            with gr.Column():
                                new_object_button = gr.Button(
                                    value="Add New Object", 
                                    interactive=True
                                )
                        track_for_video = gr.Button(
                            value="Start Tracking",
                                interactive=True,
                                )
                        reset_button = gr.Button(
                            value="Reset",
                            interactive=True, visible=False,
                        )

            with gr.Column(scale=0.5):
                output_video = gr.Video(label='Visualize Results', elem_id="input_output_video")
                output_mp4 = gr.File(label="Predicted video")
                output_mask = gr.File(label="Predicted masks")

        gr.Markdown(
            '''
            <div style="text-align:center; margin-top: 20px;">
                The authors of this work highly appreciate Meta AI for making SAM2 publicly available to the community. 
                The interface was built on <a href="https://github.com/z-x-yang/Segment-and-Track-Anything/blob/main/tutorial/tutorial%20for%20WebUI-1.0-Version.md" target="_blank">SegTracker</a>, which is also an amazing tool for video segmentation tracking. 
                <a href="https://docs.google.com/document/d/1idDBV0faOjdjVs-iAHr0uSrw_9_ZzLGrUI2FEdK-lso/edit?usp=sharing" target="_blank">Data source</a>
            </div>
                '''
        )

    ##########################################################
    ######################  back-end #########################
    ##########################################################

        # listen to the preprocess button click to get the first frame of video with scaling
        preprocess_button.click(
            fn=handle_get_meta_from_video,
            inputs=[
                session_id,
                input_video,
                scale_slider,
                config_dropdown,
                checkpoint_dropdown
            ],
            outputs=[
                input_first_frame, drawing_board, frame_per, slider_state, output_video, output_mp4, output_mask, ann_obj_id, max_obj_id, obj_id_slider
            ], queue=False, every=15
        )

        frame_per.release(
            fn=handle_show_res_by_slider, 
            inputs=[
                session_id, frame_per, slider_state, click_stack
                ], 
            outputs=[
                input_first_frame, drawing_board, frame_num
            ]
        )

        # Interactively modify the mask acc click
        input_first_frame.select(
            fn=handle_sam_click,
            inputs=[
                session_id, frame_num, point_mode, click_stack, ann_obj_id
            ],
            outputs=[
                input_first_frame, drawing_board, click_stack
            ]
        )

        # Track object in video
        track_for_video.click(
            fn=handle_tracking_objects,
            inputs=[
                session_id,
                frame_num,
                input_video,
            ],
            outputs=[
                input_first_frame,
                drawing_board,
                output_video,
                output_mp4,
                output_mask,
                click_stack
            ], queue=False, every=15
        )

        reset_button.click(
            fn=handle_reset,
            inputs=[session_id],
            outputs=[
                click_stack, input_first_frame, drawing_board, frame_per, slider_state, output_video, output_mp4, output_mask, ann_obj_id, max_obj_id, obj_id_slider
            ]
        )

        new_object_button.click(
            fn=handle_increment_ann_obj_id, 
            inputs=[ session_id, max_obj_id ], 
            outputs=[ ann_obj_id, max_obj_id, obj_id_slider ]
        )

        obj_id_slider.change(
            fn=handle_update_current_id, 
            inputs=[session_id, obj_id_slider], 
            outputs={ann_obj_id}
        )

        tab_stroke.select(
            fn=handle_drawing_board_get_input_first_frame,
            inputs=[session_id, input_first_frame],
            outputs=[drawing_board,],
        )

        seg_acc_stroke.click(
            fn=handle_sam_stroke,
            inputs=[
                session_id, drawing_board, last_draw, frame_num, ann_obj_id
            ],
            outputs=[
                input_first_frame, drawing_board, last_draw
            ]
        )

        input_video.change(
            fn=handle_extract_video_info,
            inputs=[session_id, input_video],
            outputs=[scale_slider, frame_per, slider_state, input_first_frame, drawing_board, output_video, output_mp4, output_mask], queue=False, every=15
        )
        
    app.queue(concurrency_count=1)
    app.launch(debug=True, enable_queue=True, share=False, server_name="0.0.0.0", server_port=18862)
    # app.launch(debug=True, enable_queue=True, share=True)

if __name__ == "__main__":
    mp.set_start_method("spawn")
    monitor_thread = threading.Thread(target=monitor_and_cleanup_processes)
    monitor_thread.daemon = True
    monitor_thread.start()
    seg_track_app()