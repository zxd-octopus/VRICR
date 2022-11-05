# -*- coding: utf-8 -*-


"""
This file contains several useful functions
"""
import time
import torch
import numpy as np
import argparse
import functools
import scipy.sparse as sparse


def sparse_matrix_normal_row(s_matrix):
    x_len, y_len = s_matrix.shape
    a = s_matrix.sum(1)
    a += 0.001
    ones = sparse.csr_matrix(np.ones((x_len, 1), dtype=np.float))
    b = ones / a
    normal_s_matrix = sparse.csr_matrix(s_matrix.multiply(b))
    return normal_s_matrix


def sequence_mask(lengths, max_len=None):
    if max_len is None:
        max_len = lengths.max().item()
    mask = torch.arange(0, max_len, dtype=torch.long, device=lengths.device).type_as(lengths)
    mask = mask.unsqueeze(0)
    mask = mask.repeat(1, *lengths.size(), 1)
    mask = mask.squeeze(0)
    mask = mask.lt(lengths.unsqueeze(-1))
    return mask


def reverse_sequence_mask(lengths, max_len=None):
    if max_len is None:
        max_len = lengths.max().item()
    mask = torch.arange(0, max_len, dtype=torch.long, device=lengths.device).type_as(lengths)
    mask = mask.unsqueeze(0)
    mask = mask.repeat(1, *lengths.size(), 1)
    mask = mask.squeeze(0)
    mask = mask.ge(lengths.unsqueeze(-1))
    return mask


def max_lens(X):
    """
    max_lens
    """
    if not isinstance(X[0], list):
        return [len(X)]
    elif not isinstance(X[0][0], list):
        return [len(X), max(len(x) for x in X)]
    elif not isinstance(X[0][0][0], list):
        return [len(X), max(len(x) for x in X),
                max(len(x) for xs in X for x in xs)]
    else:
        raise ValueError(
            "Data list whose dim is greater than 3 is not supported!")


def list2tensor(X):
    """
    list2tensor
    """
    size = max_lens(X)
    if len(size) == 1:
        tensor = torch.tensor(X)
        return tensor

    tensor = torch.zeros(size, dtype=torch.long)
    lengths = torch.zeros(size[:-1], dtype=torch.long)

    if len(size) == 2:
        for i, x in enumerate(X):
            l = len(x)
            tensor[i, :l] = torch.tensor(x)
            lengths[i] = l
    else:
        for i, xs in enumerate(X):
            for j, x in enumerate(xs):
                l = len(x)
                tensor[i, j, :l] = torch.tensor(x)
                lengths[i, j] = l

    return tensor, lengths


def one_hot_scatter(indice, num_classes):
    placeholder = indice.new_zeros(indice.size(0), num_classes)
    placeholder.scatter_(1, indice.reshape(-1, 1), 1.)
    return placeholder


def one_hot_sign(indice, num_classes):
    one_hot_tag = one_hot_scatter(indice, num_classes + 1).long()
    sign = torch.ones_like(one_hot_tag) - torch.cumsum(one_hot_tag, 1)
    sign = sign[:, :-1]
    return sign


def one_hot(indice, num_classes):
    """one_hot

    not efficient
    """
    I = torch.eye(num_classes).to(indice.device)
    T = I[indice]
    return T


def str2bool(v):
    """
    str2bool
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def index_select(condidate_ids, needed_id):
    matching_tag = (condidate_ids == needed_id).long()
    select_batch_size = matching_tag.sum()
    if select_batch_size < 1:
        return None
    _, index = matching_tag.sort()
    select_index = index[- select_batch_size:]
    return select_index


def nested_index_select(origin_data, select_index):
    origin_data_shape = list(origin_data.shape)
    select_index_shape = list(select_index.shape)
    work_axes = len(select_index_shape) - 1
    grad_v = functools.reduce(lambda x, y: x * y, origin_data_shape[:work_axes])
    new_dim = select_index_shape[-1]
    grad = torch.arange(0, grad_v, dtype=torch.long, device=origin_data.device).unsqueeze(-1)
    grad = grad.expand(-1, new_dim)
    grad = grad.reshape(-1)
    grad = grad * origin_data_shape[work_axes]
    select_index = select_index.reshape(-1) + grad
    reshaped_data = origin_data.reshape(grad_v * origin_data_shape[work_axes], -1)
    selected_data = reshaped_data.index_select(0, select_index)
    origin_data_shape[work_axes] = new_dim
    selected_data = selected_data.reshape(origin_data_shape)
    return selected_data


def select_value(tensor_list, selected_index):
    return [h.index_select(0, selected_index) if h is not None else h for h in tensor_list]


def strftime():
    return time.strftime("%Y-%m-%d-%H-%M-%S")


def adjust_learning_rate(optimizer,
                         rate_decay,
                         mini_lr):
    """Sets the learning rate decay"""
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        param_group['lr'] = max(lr * rate_decay, mini_lr)


def partition_arg_topK(matrix, K, axis=0, select_high=True):
    """
    perform topK based on np.argpartition
    Args:
        matrix: to be sorted
        K: select and sort the top K items
        axis: 0 or 1. dimension to be sorted.
        select_high: default is True, select top-k high value

    Returns:

    """
    if select_high:
        matrix = matrix * -1

    a_part = np.argpartition(matrix, K, axis=axis)

    if axis == 0:
        row_index = np.arange(matrix.shape[1 - axis])
        a_sec_argsort_K = np.argsort(matrix[a_part[0:K, :], row_index], axis=axis)
        return a_part[0:K, :][a_sec_argsort_K, row_index]
    else:
        column_index = np.arange(matrix.shape[1 - axis])[:, None]
        a_sec_argsort_K = np.argsort(matrix[column_index, a_part[:, 0:K]], axis=axis)
        return a_part[:, 0:K][column_index, a_sec_argsort_K]


if __name__ == '__main__':
    a = torch.tensor([[[-0.1643, -0.2033, 1.6442],
                       [-0.3004, -0.5772, 0.2693],
                       [-0.5199, 0.2751, 0.2825]],

                      [[-0.1841, 1.1484, -1.3583],
                       [-0.5199, 0.2751, 0.2825],
                       [1,3,4]],

                      [[2.2097, -1.1256, 1.7702],
                       [0.1007, -0.1140, -1.0918],
                       [2,4,6]]])
    b = torch.tensor([[0,1],
                      [1,2],
                      [0,2]], dtype=torch.long)
    print(nested_index_select(a, b))
    # tensor([[[-0.1643, -0.2033,  1.6442]],
    #
    #         [[-0.5199,  0.2751,  0.2825]],
    #
    #         [[ 2.2097, -1.1256,  1.7702]]])
    b = np.random.randn(3, 4)
    idx = partition_arg_topK(b, 2, axis=1, select_high=True)
    print(b)
    print(idx)

    a = torch.tensor([[4, 1, 2], [3, 4, 5]], dtype=torch.float)
    b = torch.tensor([[2, 2, 1], [2, 1, 1]], dtype=torch.long)
    c = torch.zeros(2, 5)
    print(c.scatter(1, b, a))
    print(c.scatter_add(1, b, a))
