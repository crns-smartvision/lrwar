
import os.path as osp
import glob
import torch
import random
# import librosa
import numpy as np
from src.utils.utils import read_txt_lines
from src.utils.utils import read_annotation_from_csv_file
from torch.utils.data import Dataset
from src.utils.preprocess import Compose, Normalize, RandomCrop, CenterCrop, HorizontalFlip


# accepted modalities for LRWAR-Landmarks dataset
MODALITIES = ["all", "landmarks", "video"]

class LRWARLandmarks(Dataset):
    def __init__(self, directory, data_partition = "train", modality = "all", landmarks_subset_idx = None, labels_subset_file="sorted_labels.txt", preprocessing_configs=None, filtered_ids_file=None):
        self.root = directory
        self.split = data_partition
        self.modality = modality
        self.landmarks_subset_idx = landmarks_subset_idx
        self.labels_subset_file = labels_subset_file
        self.filtered_instance_ids = read_txt_lines(osp.join(self.root, filtered_ids_file)) if filtered_ids_file is not None else []
        self.preprocessing = None
        if preprocessing_configs is not None:
            self.parse_preprocessing_configs(preprocessing_configs, data_partition)

        self.fps = 25

        self.video_files = []
        self.landmarks_files = []
        self.annotation_files = []
        self.labels = [] # this will gather all labels corresponding to the video files

        assert self.modality in MODALITIES, "Modality not supported, choose from {}".format(MODALITIES)
        self.partition_dir = osp.join(self.root, self.split)
        self.labels_file = osp.join(self.root, self.labels_subset_file)

        assert osp.exists(self.labels_file) and \
               osp.exists(self.partition_dir) and \
               osp.exists(self.root), \
            "LRW-AR Landmarks dataset is not complete, check the root path or the folder structure"

        # If we are not using the full set of labels, remove examples for labels not used
        # self._labels gather all labels included in the dataset
        self._labels = read_txt_lines(self.labels_file)
        self.number_classes = len(self._labels)
        self.get_list_data_paths()

    def get_list_data_paths(self):
        convert_annotation_to_video = lambda x: x.replace("annotation_", "video_").replace(".csv", ".npz")
        convert_video_2_annotation = lambda x: x.replace("video_", "annotation_").replace(".npz", ".csv")
        get_labels_from_annotation_csv_file = lambda x, label: read_annotation_from_csv_file(x, label)

        # Assuming video files are always present
        for label in self._labels:
            current_video_files = glob.glob(
                osp.join(self.root, self.split) + osp.sep + label + osp.sep + "video_*.npz")

            # remove any video file if the basename is in the list of filtered basenames
            current_video_files = [f for f in current_video_files if get_instance_id(f) not in self.filtered_instance_ids]
            current_annotation_files = [convert_video_2_annotation(f) for f in current_video_files]
            # check if the annotation file is present and valid
            assert all([osp.exists(f) for f in current_annotation_files]), "Some annotation files are missing"
            filtered_annotation_files = [f for f in current_annotation_files if len(get_labels_from_annotation_csv_file(f, get_label_from_file_path(f))) == 1]

            # update the current video files
            filtered_video_files = [convert_annotation_to_video(f) for f in filtered_annotation_files]
            self.video_files.extend(filtered_video_files)

            self.labels.extend([label for f in filtered_video_files])
            self.annotation_files.extend(filtered_annotation_files)

        assert len(self.video_files) == len(self.annotation_files), "Some annotation files are missing"

        if self.modality == "all" or self.modality == "landmarks":
            for video_file in self.video_files:
                landmarks_file = video_file.replace("video_", "landmarks_")
                assert osp.exists(landmarks_file), f"Landmark file {landmarks_file} is missing"
                self.landmarks_files.append(landmarks_file)
            assert len(self.video_files) == len(self.landmarks_files), "Some landmarks files are missing"

    def parse_preprocessing_configs(self, preprocessing_configs, data_partition):
        assert data_partition in preprocessing_configs.keys(), "Data partition {} not found in preprocessing configs".format(data_partition)
        key = data_partition
        value = preprocessing_configs[key]

        self.preprocessing = None
        composed_transforms = []

        for transform in value:
            if transform == "normalize":
                if preprocessing_configs[key][transform]["enabled"]:
                    mean = preprocessing_configs[key][transform]["mean"]
                    std = preprocessing_configs[key][transform]["std"]
                    composed_transforms.append(Normalize(mean = mean, std = std))

            elif transform == "random_crop":
                if preprocessing_configs[key][transform]["enabled"]:
                    wind_size = preprocessing_configs[key][transform]["size"]
                    composed_transforms.append(RandomCrop((wind_size, wind_size)))

            elif transform == "central_crop":
                if preprocessing_configs[key][transform]["enabled"]:
                    wind_size = preprocessing_configs[key][transform]["size"]
                    composed_transforms.append(CenterCrop((wind_size, wind_size)))

            elif transform == "random_horizontal_flip":
                if preprocessing_configs[key][transform]["enabled"]:
                    prob = preprocessing_configs[key][transform]["probability"]
                    composed_transforms.append(HorizontalFlip(prob))
            else:
                raise NotImplementedError("Transform {} not supported".format(transform))

        self.preprocessing = Compose(composed_transforms)
        return



    def __len__(self):
        return len(self.annotation_files)
    def __getitem__(self, item):
        raw_video_data = None
        raw_landmarks_data = None

        # get label of the item and the corresponding start and stop instances from the annotation file
        annotation_file = self.annotation_files[item]
        current_label = self.labels[item]
        annotation = read_annotation_from_csv_file(annotation_file, current_label)
        assert len(annotation) == 1, f"Only one instance of annotation is supported instance_ids {get_instance_id(annotation_file)}"

        if self.modality == "all":
            video_file = self.video_files[item]
            raw_video_data = self.load_video_data(video_file)

            landmarks_file = self.landmarks_files[item]
            raw_landmarks_data = self.load_landmarks_data(landmarks_file)
            assert raw_video_data.shape[0] == raw_landmarks_data.shape[0], "Video and landmarks data should have the same number of frames"

        elif self.modality == "landmarks":
            landmarks_file = self.landmarks_files[item]
            raw_landmarks_data = self.load_landmarks_data(landmarks_file)

        elif self.modality == "video":
            video_file = self.video_files[item]
            raw_video_data = self.load_video_data(video_file)

        else:
            raise NotImplementedError("Modality not supported, choose from {}".format(MODALITIES))

        if self.split == 'train' and len(annotation)==1:
            video_data, landmarks_data = self._apply_variable_length_aug(annotation[0],
                                                   raw_video_data = raw_video_data,
                                                   raw_landmarks_data= raw_landmarks_data)
            video_data = self.preprocessing(video_data)
        else:
            video_data = self.preprocessing(raw_video_data)
            landmarks_data = raw_landmarks_data

        if self.modality == "all":
            return video_data, landmarks_data, self._labels.index(current_label)
        elif self.modality == "landmarks":
            return landmarks_data, self._labels.index(current_label)
        elif self.modality == "video":
            return video_data, self._labels.index(current_label)

    def _apply_variable_length_aug(self, annotation, raw_video_data = None, raw_landmarks_data = None):
        assert raw_landmarks_data is not None or raw_video_data is not None, "Either video or landmarks data should be provided"

        start = annotation["start"]
        end = annotation["end"]

        utterance_duration = float(end) - float(start)
        half_interval = int(utterance_duration / 2.0 * self.fps)  # num frames of utterance / 2
        if raw_video_data is not None:
            n_frames = raw_video_data.shape[0]
        else:
            n_frames = raw_landmarks_data.shape[0]

        mid_idx = (n_frames - 1) // 2  # video has n frames, mid point is (n-1)//2 as count starts with 0
        left_idx = random.randint(0, max(0, mid_idx - half_interval - 1))  # random.randint(a,b) chooses in [a,b]
        right_idx = random.randint(min(mid_idx + half_interval + 1, n_frames), n_frames)

        aug_video_data = None if raw_video_data is None else raw_video_data[left_idx:right_idx]
        aug_landmarks_data = None if raw_landmarks_data is None else raw_landmarks_data[left_idx:right_idx]

        return aug_video_data, aug_landmarks_data


    def load_landmarks_data(self, filename):
        landmarks = np.load(filename)['data']
        if self.landmarks_subset_idx is not None:
            landmarks = landmarks[:, self.landmarks_subset_idx,:]
        else:
            landmarks = landmarks
        return landmarks

    def load_video_data(self, filename):
        assert filename.endswith('npz'), "Video file should be in npz format"
        return np.load(filename)['data']

def get_instance_id(file_path):
    ids = osp.basename(file_path).split(".")[0].split("_")
    instance_id = "_".join(ids[-2:])
    return instance_id

def get_label_from_file_path(file_path):
    ids = osp.basename(file_path).split(".")[0].split("_")
    return ids[-1]

def pad_packed_collate(batch):
    if len(batch) == 1:
        data, lengths, labels_np, landmarks = zip(
            *[(a, a.shape[0], b, c) for (a, b, c) in sorted(batch, key=lambda x: x[0].shape[0], reverse=True)])
        data = torch.FloatTensor(data)
        lengths = [data.size(1)]
        landmarks = torch.FloatTensor(landmarks)

    if len(batch) > 1:
        data_list, lengths, labels_np, landmarks = zip(
            *[(a, a.shape[0], b, c) for (a, b, c) in sorted(batch, key=lambda x: x[0].shape[0], reverse=True)])

        if data_list[0].ndim == 3:
            max_len, h, w = data_list[0].shape  # since it is sorted, the longest video is the first one
            data_np = np.zeros((len(data_list), max_len, h, w))
        elif data_list[0].ndim == 1:
            max_len = data_list[0].shape[0]
            data_np = np.zeros((len(data_list), max_len))
        for idx in range(len(data_np)):
            data_np[idx][:data_list[idx].shape[0]] = data_list[idx]
        data = torch.FloatTensor(data_np)

        # todo padding to get same seq length for one batch
        if landmarks[0].ndim == 3:
            max_len, h, w = landmarks[0].shape  # since it is sorted, the longest video is the first one
            landmarks_np = np.zeros((len(landmarks), max_len, h, w))
        elif data_list[0].ndim == 1:
            max_len = landmarks[0].shape[0]
            landmarks_np = np.zeros((len(landmarks), max_len))
        for idx in range(len(landmarks_np)):
            landmarks_np[idx][:landmarks[idx].shape[0]] = landmarks[idx]
        landmarks = torch.FloatTensor(landmarks_np)

    labels = torch.LongTensor(labels_np)
    return data, lengths, labels, landmarks

class LRWAR(Dataset):
    def __init__(self, directory, data_partition = "train"):
        self.root = directory
        self.split = data_partition
        self.partition_dir = osp.join(self.root, self.split)
        self.labels_file = osp.join(self.root, "sorted_classes.txt")

        assert osp.exists(self.labels_file) and \
               osp.exists(self.partition_dir) and \
               osp.exists(self.root), \
            "LRW-AR dataset is not complete, check the root path or the folder structure"

        # If we are not using the full set of labels, remove examples for labels not used
        self._labels = read_txt_lines(self.labels_file)
        self.number_classes = len(self._labels)
        self.get_list_data_paths()

    def get_list_data_paths(self):
        self.video_files = []
        self.annotation_files = []
        for label in self._labels:
            current_video_files = glob.glob(osp.join(self.root, self.split) + osp.sep + label + osp.sep+"*.mp4")
            current_annotation_files = [f.replace(".mp4", ".csv") for f in current_video_files]
            assert all([osp.exists(f) for f in current_annotation_files]), "Some annotation files are missing"
            self.video_files.extend(current_video_files)
            self.annotation_files.extend(current_annotation_files)


    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, item):
        # TODO implement get item for LRWAR
        pass





