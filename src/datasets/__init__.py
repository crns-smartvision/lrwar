from src.datasets.lrwar import LRWAR, LRWARLandmarks

DATA_PARTITIONS = ['train', 'val', 'test']

def dataset_factory(cfg, data_partition):
    assert data_partition in DATA_PARTITIONS, f"Data partition {data_partition} not recognized. It must be one of {DATA_PARTITIONS}"
    dataset_name = cfg.name
    dataset_directory = cfg.dir

    if dataset_name == 'lrw-ar':
        return LRWAR(dataset_directory)
    elif dataset_name == 'lrw-ar-landmarks':
        landmaks_subset_idx = None if not "landmarks_subset_idx" in cfg.params else cfg.params.landmarks_subset_idx
        preprocessing = None if not "preprocessing" in cfg.params else cfg.params.preprocessing
        filtered_ids_file = None if not "filtered_ids_file" in cfg.params else cfg.params.filtered_ids_file
        labels_subset_file = cfg.params.labels_subset_file
        return LRWARLandmarks(dataset_directory, data_partition = data_partition,
                              labels_subset_file=labels_subset_file,
                              landmarks_subset_idx=landmaks_subset_idx,
                              preprocessing_configs=preprocessing,
                              filtered_ids_file=filtered_ids_file)
    else:
        raise ValueError(f"Dataset {dataset_name} not recognized")
