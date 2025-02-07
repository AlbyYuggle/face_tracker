from PIL import Image
import cv2
import matplotlib.pyplot as plt
from deepface import DeepFace
import numpy as np
import os
import argparse

def expand_bb(x, y, w, h, expand_ratio=0.25):
    
    w_expand = w * expand_ratio
    h_expand = h * expand_ratio
    
    new_x = x - w_expand
    new_y = y - h_expand
    new_w = w + (2 * w_expand)
    new_h = h + (2 * h_expand)
    
    return int(new_x), int(new_y), int(new_w), int(new_h)

def get_reference_face(image_path):
    img = np.array(Image.open(image_path))

    extraction = DeepFace.extract_faces(
        img_path = img, 
        detector_backend = "retinaface",
        align = True,
        enforce_detection = False
    )[0]

    face = extraction['face']
    confidence = extraction['confidence']
    facial_area = extraction['facial_area']
    x, y, w, h = expand_bb(facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h'])


    if confidence < 0.95:
        raise Exception("No face detected in reference image")
    
    ref_face = img[max(0,y):min(y+h,img.shape[0]), max(0,x):min(x+w,img.shape[1]),:]
    return ref_face

def pad_image_to_size(image, target_height, target_width):
    height, width, _ = image.shape
    top = (target_height - height) // 2
    bottom = target_height - height - top
    left = (target_width - width) // 2
    right = target_width - width - left
    
    padded_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return padded_image

def images_to_video(image_list, output_video_path, fps):

    height, width, _ = image_list[0].shape
    for image in image_list:
        height = max(height, image.shape[0])
        width = max(width, image.shape[1])

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    for image in image_list:
        out.write(pad_image_to_size(image, height, width))
    
    out.release()
    

def video_to_images(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened():
        return []
    
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    return frames, fps

def frame_to_timestamp(frame_no, fps):
    seconds = int(frame_no/fps)

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_clip_title(input_path, clip_no):
    filename = os.path.basename(input_path)
    name, ext = filename.split('.')
    return name + "_" + str(clip_no) + "." + ext

def setup_args():
   
   parser = argparse.ArgumentParser()
   
   parser.add_argument('-i', '--input_path', 
                       required=True,
                       help='Path to input file')
   
   parser.add_argument('-o', '--output_dir',
                       required=True, 
                       help='Directory path for output files')
   
   parser.add_argument('-r', '--ref_path',
                       required=True,
                       help='Path to reference image')

   args = parser.parse_args()
   return args

def IOU(bb1, bb2):

    x1, y1, w1, h1 = bb1
    x2, y2, w2, h2 = bb2
    
    bb1_x2 = x1 + w1
    bb1_y2 = y1 + h1
    bb2_x2 = x2 + w2
    bb2_y2 = y2 + h2
    
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(bb1_x2, bb2_x2)
    y_bottom = min(bb1_y2, bb2_y2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    bb1_area = w1 * h1
    bb2_area = w2 * h2
    
    union_area = bb1_area + bb2_area - intersection_area
    
    iou = intersection_area / union_area
    
    return iou