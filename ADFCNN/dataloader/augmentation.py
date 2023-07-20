import numpy as np


def cutcat(data_1, label_1, data_2, label_2, num_classes, ratio=8):
    c, t = 22, 751
    
    length = np.random.randint(t // ratio)
    
    x = np.random.randint(t)
    x1 = np.clip(x - length // 2, 0, t)
    x2 = np.clip(x + length // 2, 0, t)
    
    mask_1 = np.ones((c, t), np.float32)
    mask_1[:, x1:x2] = 0.
    data_1 = data_1 * mask_1
    
    mask_2 = np.zeros((c, t), np.float32)
    mask_2[:, x1:x2] = 1.
    data_2 = data_2 * mask_2
    
    data = data_1 + data_2
    
    one_hot_label_1 = np.zeros(num_classes, np.float32)
    one_hot_label_1[label_1] = 1.
    
    one_hot_label_2 = np.zeros(num_classes, np.float32)
    one_hot_label_2[label_2] = 1.
    
    lamb = (x2 - x1) / t
    label = (1 - lamb) * one_hot_label_1 + lamb * one_hot_label_2
    
    return data, label


def cutcat_2(data_1, label_1, data_2, label_2, num_classes, ratio=8):

    data = data_1
    one_hot_label_1 = np.zeros(num_classes, np.float32)
    one_hot_label_1[label_1] = 1.


    label = one_hot_label_1

    return data, label
