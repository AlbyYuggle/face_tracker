from utils import video_to_images, images_to_video, frame_to_timestamp, format_clip_title, setup_args, IOU
from deepface import DeepFace
import matplotlib.pyplot as plt
import cv2 
import json
import os
from kalman_filter import KalmanFilter
import argparse
import copy


def run(input_path, ref_path, output_dir):
    images, fps = video_to_images(input_path)

    test_frames = []
    crops = []
    bbs = []
    json_objs = []
    start_frame = 0
    clip_no = 1

    kf = KalmanFilter(2, 1, 10)

    for frame_no, img in enumerate(images):

        result = DeepFace.verify(
            img1_path = img,
            img2_path = ref_path,
            model_name = "GhostFaceNet",
            enforce_detection = False
        )

        if result['distance'] > result['threshold']:
            found_match = False

            if kf.initialized:

                kx, ky, kw, kh = kf.predict()

                x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']

                extractions = DeepFace.extract_faces(
                    img_path = img, 
                    detector_backend = "retinaface",
                    align = True,
                    enforce_detection = False
                )

                found_match = False
                face = None
                facial_area = None

                for extraction in extractions:

                    face = extraction['face']
                    facial_area = extraction['facial_area']
                    confidence = extraction['confidence']
                    if confidence >= 0.90 and x <= kx + kw//2 <= x+w and y <= ky + kh//2 <= y+h and IOU((x, y, w, h), (kx, ky, kw, kh)) >= 0.25:
                        found_match = True
                        x, y, w, h = kx, ky, kw, kh
                        break

            if found_match:

                face = img[y:y+h, x:x+w]
                crops.append(copy.deepcopy(face))
                bbs.append((x, y, w, h))

                img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                if frame_no == len(images) - 1 and len(crops) > int(fps)//2:
                    clip_title = format_clip_title(input_path, clip_no)

                    images_to_video(crops[:-1], os.path.join(output_dir, clip_title), fps)

                    start_time = frame_to_timestamp(start_frame, fps)
                    end_time = frame_to_timestamp(frame_no, fps)

                    json_objs.append({"filename": clip_title, "start": start_time, "end": end_time, "coords": bbs})

            else:
                if frame_no != start_frame:
                    
                    if len(crops) > int(fps)//4:
                        clip_title = format_clip_title(input_path, clip_no)

                        images_to_video(crops[:-1], os.path.join(output_dir, clip_title), fps)

                        start_time = frame_to_timestamp(start_frame, fps)
                        end_time = frame_to_timestamp(frame_no, fps)

                        json_objs.append({"filename": clip_title, "start": start_time, "end": end_time, "coords": bbs})
                    
                        clip_no += 1
                    crops = []
                    bbs = []
                    kf.initialized = False
                start_frame = frame_no + 1
            
            
        else:

            facial_area = result['facial_areas']['img1']
            x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
            face = img[y:y+h, x:x+w]

            if not kf.initialized:
                kf.__init__(2, 1, 10)
                kf.initialize((x, y, w, h))

            kx, ky, kw, kh = kf.predict()
            kf.update((x, y, w, h))

            crops.append(copy.deepcopy(face))
            bbs.append((x, y, w, h))

            img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            img = cv2.rectangle(img, (kx, ky), (kx + kw, ky + kh), (0, 255, 0), 2)

            if frame_no == len(images) - 1 and len(crops) > 15:
                clip_title = format_clip_title(input_path, clip_no)

                images_to_video(crops[:-1], os.path.join(output_dir, clip_title), fps)

                start_time = frame_to_timestamp(start_frame, fps)
                end_time = frame_to_timestamp(frame_no, fps)

                json_objs.append({"filename": clip_title, "start": start_time, "end": end_time, "coords": bbs})
                
        test_frames.append(img)

    images_to_video(test_frames, os.path.join(output_dir, "detections.mp4"), fps)
    with open(os.path.join(output_dir, "output.json"), "w") as f:
        json.dump(json_objs, f)

if __name__ == "__main__":
    args = setup_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    run(args.input_path, args.ref_path, args.output_dir)