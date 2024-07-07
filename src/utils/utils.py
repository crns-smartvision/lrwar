import os
import math
import numpy as np
import csv
import hydra

def read_txt_lines(filepath):
    assert os.path.isfile( filepath ), "Error when trying to read txt file, path does not exist: {}".format(filepath)
    with open( filepath ) as myfile:
        content = myfile.read().splitlines()
    return content

def save2npz(filename, data=None):
    assert data is not None, "data is {}".format(data)
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    np.savez_compressed(filename, data=data)


def read_annotation_from_csv_file(filepath, label=None):
    annotations = []

    with open(filepath) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        #,encodings="utf-8-sig")
        for row in csv_reader:
            annotations.append({"label": row[3], "start": row[2], "end": row[1]}) # row[0] is the confidence but not used in our case.

    if annotations and label is not None:
        return [a for a in annotations if a["label"] == label]
    else:
        return annotations

def read_csv_lines(filepath):
    #assert os.path.isfile( filepath ), "Error when trying to read csv file, path does not exist: {}".format(filepath)
    with open(filepath) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        #,encodings="utf-8-sig")
        for row in csv_reader:
            if row[3]== (os.path.split(os.path.split(filepath)[0])[1]):
                start = row[2]
                end = row[1]
                print(start)
                print(end)
                csv_file.close()
                return start, end

def change_lr_on_optimizer(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class CosineScheduler:
    def __init__(self, lr_ori, epochs):
        self.lr_ori = lr_ori
        self.epochs = epochs

    def adjust_lr(self, optimizer, epoch):
        reduction_ratio = 0.5 * (1 + math.cos(math.pi * epoch / self.epochs))
        change_lr_on_optimizer(optimizer, self.lr_ori*reduction_ratio)


def get_cwd():
    """
    custom function to get the current hydra output directory while keeping the original working directory
    """
    original_cwd = hydra.utils.get_original_cwd()
    cwd_dir = os.getcwd()
    os.chdir(original_cwd)
    return cwd_dir

