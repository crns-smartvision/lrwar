import torch
import numpy as np



def pad_packed_collate(batch):
    if len(batch) == 1:
        data, lengths, labels_np, landmarks = zip(*[(a, a.shape[0], b, c) for (a, b,c) in sorted(batch, key=lambda x: x[0].shape[0], reverse=True)])
        data = torch.FloatTensor(data)
        lengths = [data.size(1)]
        landmarks = torch.FloatTensor(landmarks)

    if len(batch) > 1:
        data_list, lengths, landmarks, labels_np= zip(*[(a, a.shape[0], b, c) for (a, b, c) in sorted(batch, key=lambda x: x[0].shape[0], reverse=True)])

        if data_list[0].ndim == 3:
            max_len, h, w = data_list[0].shape  # since it is sorted, the longest video is the first one
            data_np = np.zeros((len(data_list), max_len, h, w))
        elif data_list[0].ndim == 1:
            max_len = data_list[0].shape[0]
            data_np = np.zeros((len(data_list), max_len))
        for idx in range(len(data_np)):
            data_np[idx][:data_list[idx].shape[0]] = data_list[idx]
        data = torch.FloatTensor(data_np)

        #todo padding to get same seq length for one batch
        if landmarks[0].ndim == 3:
            max_len, h, w = landmarks[0].shape  # since it is sorted, the longest video is the first one
            landmarks_np = np.zeros((len(landmarks), max_len, h, w))
        elif data_list[0].ndim == 1:
            max_len = landmarks[0].shape[0]
            landmarks_np = np.zeros((len(landmarks), max_len))
        for idx in range( len(landmarks_np)):
            landmarks_np[idx][:landmarks[idx].shape[0]] = landmarks[idx]
        landmarks = torch.FloatTensor(landmarks_np)

    labels = torch.LongTensor(labels_np)
    return data, lengths, labels, landmarks


# -- mixup data augmentation, adapted from
# from https://github.com/hongyi-zhang/mixup/blob/master/cifar/utils.py
def mixup_data(x, y,landmarks, alpha=1.0, soft_labels = None, use_cuda=False):
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''

    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index,:]
    mixed_landmarks = lam * landmarks + (1 - lam) * landmarks[index,:]

    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam, mixed_landmarks


def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
