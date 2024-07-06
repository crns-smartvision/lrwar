import os
import os.path as osp
from collections import deque

import glob
import cv2
import numpy as np

import logging
import hydra
import json
from omegaconf import OmegaConf
import warnings

import dlib
import mediapipe as mp

from _version import __version__

from src.utils.transform import linear_interpolate, warp_img, apply_transform, cut_patch, convert_bgr2gray
from src.utils.transform import extract_landmarks_mediapipe, extract_landmarks_dlib
from src.utils.utils import save2npz, get_cwd

logger = logging.getLogger("Training Data generation")
logger.setLevel(logging.INFO)
warnings.filterwarnings("ignore")


# -- mean face utils
STD_SIZE = (256, 256)
stablePntsIDs = [33, 36, 39, 42, 45]

def landmarks_interpolate(landmarks):
    """Interpolate landmarks
    param list landmarks: landmarks detected in raw videos
    """

    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    if not valid_frames_idx:
        return None
    for idx in range(1, len(valid_frames_idx)):
        if valid_frames_idx[idx] - valid_frames_idx[idx - 1] == 1:
            continue
        else:
            landmarks = linear_interpolate(landmarks, valid_frames_idx[idx - 1], valid_frames_idx[idx])
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    # -- Corner case: keep frames at the beginning or at the end failed to be detected.
    if valid_frames_idx:
        landmarks[:valid_frames_idx[0]] = [landmarks[valid_frames_idx[0]]] * valid_frames_idx[0]
        landmarks[valid_frames_idx[-1]:] = [landmarks[valid_frames_idx[-1]]] * (
                    len(landmarks) - valid_frames_idx[-1])
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    assert len(valid_frames_idx) == len(landmarks), "not every frame has landmark"
    return landmarks

#TODO put all processing in a class to avoid overloading data from disk!
def crop_mouth_region(video, landmarks, crop_width, crop_height,
                      convert_to_gray,
                      interpolate_landmarks=False,
                      smoothing_landmarks_window=5,
                      start_landmarks_idx=48, stop_landmarks_idx=68,
                      mean_face_landmarks_path=None,
                      display=False):

    if mean_face_landmarks_path is not None:
        mean_face_landmarks = np.loadtxt(mean_face_landmarks_path)

    frame_cnt = 0
    cap = cv2.VideoCapture(video)
    while True:
        ret, image = cap.read()
        if ret == False:
            break
        # Convert the image color to grayscale
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if frame_cnt == 0:
            q_frame, q_landmarks = deque(), deque()
            sequence = []

        q_landmarks.append(landmarks[frame_cnt])
        q_frame.append(frame)

        if len(q_frame) == smoothing_landmarks_window:
            smoothed_landmarks = np.mean(q_landmarks, axis=0)
            cur_landmarks = q_landmarks.popleft()
            cur_frame = q_frame.popleft()
            # -- affine transformation
            trans_frame, trans = warp_img(smoothed_landmarks[stablePntsIDs, :],
                                          mean_face_landmarks[stablePntsIDs, :],
                                          cur_frame,
                                          STD_SIZE)

            trans_landmarks = trans(cur_landmarks)

            if display:
                cv2.imshow("image",cut_patch( trans_frame,
                                            trans_landmarks[start_landmarks_idx:stop_landmarks_idx],
                                            crop_height//2,
                                            crop_width//2,))
                key = cv2.waitKey(100)
                if key == 27:
                    cv2.destroyAllWindows()
                    break

            # -- crop mouth patch
            sequence.append(cut_patch(trans_frame,
                                      trans_landmarks[start_landmarks_idx:stop_landmarks_idx],
                                      crop_height // 2,
                                      crop_width // 2, ))
        if frame_cnt == len(landmarks) - 1:
            while q_frame:
                cur_frame = q_frame.popleft()
                # -- transform frame
                trans_frame = apply_transform(trans, cur_frame, STD_SIZE)
                # -- transform landmarks
                trans_landmarks = trans(q_landmarks.popleft())
                # -- crop mouth patch
                sequence.append(cut_patch(trans_frame,
                                          trans_landmarks[start_landmarks_idx:stop_landmarks_idx],
                                          crop_height // 2,
                                          crop_width // 2, ))
            return np.array(sequence)
        frame_cnt += 1
    return None


    lines = open(args.filename_path).read().splitlines()
    lines = list(filter(lambda x: 'test' == x.split('/')[-2], lines)) if args.testset_only else lines
    not_processed = 0
    for filename_idx, line in enumerate(lines):

        filename = os.path.splitext(line)[0]
        print('idx: {} \tProcessing.\t{}'.format(filename_idx, filename))

        video_pathname = os.path.join(args.video_direc, filename + '.mp4')
        landmarks_pathname = os.path.join(args.landmark_direc, filename + '.npz')
        dst_pathname = os.path.join(args.save_direc, filename + '.npz')

        assert os.path.isfile(video_pathname), "File does not exist. Path input: {}".format(video_pathname)
        assert os.path.isfile(landmarks_pathname), "File does not exist. Path input: {}".format(landmarks_pathname)

        if os.path.exists(dst_pathname):
            continue

        multi_sub_landmarks = np.load(landmarks_pathname, allow_pickle=True)['data']
        landmarks = [None] * len(multi_sub_landmarks)
        if len(landmarks) < 29:
            not_processed += 1
            print("number of sample not_processed: ", not_processed)
            continue
        for frame_idx in range(len(landmarks)):
            try:
                landmarks[frame_idx] = multi_sub_landmarks[frame_idx]
            except IndexError:
                continue

        # -- pre-process landmarks: interpolate frames not being detected.
        preprocessed_landmarks = landmarks_interpolate(landmarks)
        if not preprocessed_landmarks:
            continue

        # -- crop
        sequence = crop_patch(video_pathname, preprocessed_landmarks)
        assert sequence is not None, "cannot crop from {}.".format(filename)

        # -- save
        data = convert_bgr2gray(sequence) if args.convert_gray else sequence[..., ::-1]

        save2npz(dst_pathname, data=data)

    print('Done.')
    return None


@hydra.main(version_base=None, config_path="config", config_name="config_generate_training_data")
def main(cfg):
    logger.info("Version: " + __version__)
    dict_cfg = OmegaConf.to_container(cfg)
    cfg_pprint = json.dumps(dict_cfg, indent=4)
    logger.info(cfg_pprint)

    output_dir = get_cwd()
    logger.info(f"Working dir: {os.getcwd()}")
    logger.info(f"Export dir: {output_dir}")
    logger.info("Loading parameters from config file")

    # get the name of the dataset
    dataset_name = cfg.dataset.name
    export_dir = cfg.export_dir
    if not osp.exists(export_dir):
        os.makedirs(export_dir)

    dataset_dir = cfg.dataset.dir
    partitions = cfg.dataset.partitions

    do_mouth_croping = "mouth_croping" in cfg
    if do_mouth_croping:
        video_affix = cfg.mouth_croping.video_affix
        video_extension = cfg.mouth_croping.video_extension
        crop_width = cfg.mouth_croping.crop_width
        crop_height = cfg.mouth_croping.crop_height
        convert_to_gray = cfg.mouth_croping.convert_to_gray
        interpolate_landmarks = cfg.mouth_croping.interpolate_landmarks

    landmarks_affix = cfg.landmarks_extraction.landmarks_affix
    landmarks_extension = cfg.landmarks_extraction.landmarks_extension

    do_annotation_export = "annotation_export_parameters" in cfg
    if "annotation_export_parameters" in cfg:
        annotation_affix = cfg.annotation_export_parameters.annotation_affix
        annotation_extension = cfg.annotation_export_parameters.annotation_extension

    landmarks_detector = cfg.landmarks_extraction.landmarks_detector.name
    assert landmarks_detector == "dlib", "Only dlib is supported for now"
    if landmarks_detector == "dlib":
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(cfg.landmarks_extraction.landmarks_detector.model_path)
    elif landmarks_detector == "mediapipe":
        mp_face_detection = mp.solutions.face_detection
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_face_mesh = mp.solutions.face_mesh

    for partition in partitions:
        partition_dir = osp.join(dataset_dir, partition)
        assert osp.exists(partition_dir), f"Partition {partition} does not exist in {dataset_dir}"

        export_partition_dir = osp.join(export_dir, partition)
        if not osp.exists(export_partition_dir):
            os.makedirs(export_partition_dir)

        # go over all classes
        classes = os.listdir(partition_dir)
        for class_name in classes:
            class_dir = osp.join(partition_dir, class_name)
            assert osp.exists(class_dir), f"Class {class_name} does not exist in {partition_dir}"

            export_class_dir = osp.join(export_partition_dir, class_name)
            if not osp.exists(export_class_dir):
                os.makedirs(export_class_dir)

            # get list of videos
            videos = glob.glob(osp.join(class_dir, f"*.mp4"))

            for video in videos:
                video_id = osp.splitext(osp.basename(video))[0]
                save_path = osp.join(export_class_dir, landmarks_affix + video_id + landmarks_extension)
                logger.info(f"-----------------------------------------------------------")
                logger.info(f"Processing video {video_id} from class {class_name}")
                if landmarks_detector == "dlib":
                    extracted_landmarks = extract_landmarks_dlib(video, detector, predictor, display=cfg.landmarks_extraction.display)
                elif landmarks_detector == "mediapipe":
                    extracted_landmarks = extract_landmarks_mediapipe(video, mp_face_mesh, mp_drawing, mp_drawing_styles, display=cfg.landmarks_extraction.display)
                save2npz(save_path, data=extracted_landmarks)
                logger.info(f"Landmarks extraction ... Done!")

                if do_mouth_croping:
                    cropped_sequences = crop_mouth_region(video, extracted_landmarks, crop_width,
                                      crop_height, convert_to_gray,
                                      interpolate_landmarks, smoothing_landmarks_window=5,
                                      start_landmarks_idx=48, stop_landmarks_idx=68,
                                      mean_face_landmarks_path=cfg.mouth_croping.mean_face_landmarks_path,
                                      display= cfg.mouth_croping.display)

                    save_path = osp.join(export_class_dir, video_affix + video_id + video_extension)
                    save2npz(save_path, data=cropped_sequences)
                    logger.info(f"Cropping mouth region .... Done!")

                if do_annotation_export:
                    annotation_file = osp.join(class_dir, video_id + annotation_extension)
                    assert osp.exists(annotation_file), f"Annotation file {annotation_file} does not exist"
                    save_path = osp.join(export_class_dir, annotation_affix + video_id + annotation_extension)
                    # copy annotation file
                    os.system(f"cp {annotation_file} {save_path}")
                    logger.info(f"Exported annotation file .... Done!")


                # export the list of classes in a txt file
                class_list = osp.join(export_dir, "labels.txt")
                with open(class_list, "w") as f:
                    f.write("\n".join(classes))


if __name__ == "__main__":
    main()
