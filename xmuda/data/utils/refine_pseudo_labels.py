import torch
import numpy as np
import os

from annoy import AnnoyIndex
import pickle


def refine_pseudo_labels(probs, pseudo_label, ignore_label=-100):
    """
    Reference: https://github.com/liyunsheng13/BDL/blob/master/SSL.py
    Per class, set the less confident half of labels to ignore label.
    :param probs: maximum probabilities (N,), where N is the number of 3D points
    :param pseudo_label: predicted label which had maximum probability (N,)
    :param ignore_label:
    :return:
    """
    probs, pseudo_label = torch.tensor(probs), torch.tensor(pseudo_label)
    probs = probs[torch.arange(probs.shape[0]), pseudo_label]
    for cls_idx in pseudo_label.unique():
        curr_idx = pseudo_label == cls_idx
        curr_idx = curr_idx.nonzero().squeeze(1)
        thresh = probs[curr_idx].median()
        thresh = min(thresh, 0.9)
        ignore_idx = curr_idx[probs[curr_idx] < thresh]
        pseudo_label[ignore_idx] = ignore_label
    return pseudo_label.numpy()

def entropy_combination(probs_2d, probs_3d, temperature):
    w_2d = np.exp(-np.sum(probs_2d * np.log(probs_2d), axis=1, keepdims=True)/temperature)
    w_3d = np.exp(-np.sum(probs_3d * np.log(probs_3d), axis=1, keepdims=True)/temperature)

    probs = (w_2d * probs_2d + w_3d * probs_3d) / (w_2d + w_3d)
    pl = np.argmax(probs, axis=1)
    pl = refine_pseudo_labels(probs, pl) 
    return pl

def certainty_filter_pl(probs, ignore_label=-100):
    pl = np.argmax(probs, axis=1)
    probs = probs[np.arange(pl.shape[0]), pl]

    for cls in np.unique(pl):
        cls_idx = np.nonzero(pl == cls)[0]

        thresh = min(np.median(probs[cls_idx]), 0.9)
        ignore_idx = cls_idx[probs[cls_idx] < thresh]
        pl[ignore_idx] = ignore_label

    return pl

def agreement_filter_pl(probs_2d=None, pseudo_label_2d=None, probs_3d=None, pseudo_label_3d=None, ignore_label=-100, **kwargs):
    """
    Reference: https://github.com/liyunsheng13/BDL/blob/master/SSL.py
    Per class, set the less confident half of labels to ignore label.
    :param probs: maximum probabilities (N,), where N is the number of 3D points
    :param pseudo_label: predicted label which had maximum probability (N,)
    :param ignore_label:
    :return:
    """
    pseudo_label_2d = refine_pseudo_labels(probs_2d, pseudo_label_2d, ignore_label=ignore_label) 
    pseudo_label_3d = refine_pseudo_labels(probs_3d, pseudo_label_3d, ignore_label=ignore_label) 

    agreement = pseudo_label_2d == pseudo_label_3d
    pseudo_label_2d[~agreement] = ignore_label
    pseudo_label_3d[~agreement] = ignore_label

    return pseudo_label_2d, pseudo_label_3d

def agreement_filter_pl_2(probs_2d, pseudo_label_2d, probs_3d, pseudo_label_3d, ignore_label=-100):
    """
    Reference: https://github.com/liyunsheng13/BDL/blob/master/SSL.py
    Per class, set the less confident half of labels to ignore label.
    :param probs: maximum probabilities (N,), where N is the number of 3D points
    :param pseudo_label: predicted label which had maximum probability (N,)
    :param ignore_label:
    :return:
    """
    pseudo_label_2d = refine_pseudo_labels(probs_2d, pseudo_label_2d, ignore_label=ignore_label) 
    pseudo_label_3d = refine_pseudo_labels(probs_3d, pseudo_label_3d, ignore_label=ignore_label) 

    pseudo_label_2d[(pseudo_label_2d == ignore_label) & (pseudo_label_3d != ignore_label)] = pseudo_label_3d[(pseudo_label_2d == ignore_label) & (pseudo_label_3d != ignore_label)]
    pseudo_label_3d[(pseudo_label_2d != ignore_label) & (pseudo_label_3d == ignore_label)] = pseudo_label_2d[(pseudo_label_2d != ignore_label) & (pseudo_label_3d == ignore_label)]

    agreement = pseudo_label_2d == pseudo_label_3d
    pseudo_label_2d[~agreement] = ignore_label
    pseudo_label_3d[~agreement] = ignore_label

    return pseudo_label_2d, pseudo_label_3d

def agreement_filter_soft_centroids(probs_2d=None, pseudo_label_2d=None, features_2d=None, probs_3d=None, pseudo_label_3d=None, features_3d=None, ignore_label=-100):
    """
    Reference: https://github.com/liyunsheng13/BDL/blob/master/SSL.py
    Per class, set the less confident half of labels to ignore label.
    :param probs: maximum probabilities (N,), where N is the number of 3D points
    :param pseudo_label: predicted label which had maximum probability (N,)
    :param ignore_label:
    :return:
    """
    pseudo_label_2d = refine_pseudo_labels(probs_2d, pseudo_label_2d, ignore_label=ignore_label) 
    pseudo_label_3d = refine_pseudo_labels(probs_3d, pseudo_label_3d, ignore_label=ignore_label) 

    centroids_2d = soft_centroids(probs_2d, features_2d)
    centroids_3d = soft_centroids(probs_3d, features_3d)

    closest_2d = np.argmax(cosine(features_2d, centroids_2d), axis=1)
    closest_3d = np.argmax(cosine(features_3d, centroids_3d), axis=1)

    both_valid = (pseudo_label_2d != ignore_label) & (pseudo_label_3d != ignore_label)
    one_valid = (pseudo_label_2d != ignore_label) != (pseudo_label_3d != ignore_label)
    pl_agree = pseudo_label_2d == pseudo_label_3d
    cent_agree = closest_2d == closest_3d

    pseudo_label = ignore_label * np.ones(pseudo_label_2d.shape, dtype=pseudo_label_2d.dtype)

    pseudo_label[both_valid & pl_agree] = pseudo_label_2d[both_valid & pl_agree]
    pseudo_label[both_valid & cent_agree & (pseudo_label_2d == closest_2d)] = pseudo_label_2d[both_valid & cent_agree & (pseudo_label_2d == closest_2d)]
    pseudo_label[both_valid & cent_agree & (pseudo_label_3d == closest_3d)] = pseudo_label_3d[both_valid & cent_agree & (pseudo_label_3d == closest_3d)]

    pseudo_label[~both_valid & ~one_valid & cent_agree] = closest_2d[~both_valid & ~one_valid & cent_agree]
    pseudo_label[~both_valid & one_valid & cent_agree & (pseudo_label_2d == closest_2d)] = pseudo_label_2d[~both_valid & one_valid & cent_agree & (pseudo_label_2d == closest_2d)]
    pseudo_label[~both_valid & one_valid & cent_agree & (pseudo_label_3d == closest_3d)] = pseudo_label_3d[~both_valid & one_valid & cent_agree & (pseudo_label_3d == closest_3d)]

    return pseudo_label

def agreement_filter_hard_centroids(probs_2d=None, pseudo_label_2d=None, features_2d=None, probs_3d=None, pseudo_label_3d=None, features_3d=None, ignore_label=-100):
    """
    Reference: https://github.com/liyunsheng13/BDL/blob/master/SSL.py
    Per class, set the less confident half of labels to ignore label.
    :param probs: maximum probabilities (N,), where N is the number of 3D points
    :param pseudo_label: predicted label which had maximum probability (N,)
    :param ignore_label:
    :return:
    """
    centroids_2d = hard_centroids(pseudo_label_2d, features_2d)
    centroids_3d = hard_centroids(pseudo_label_3d, features_3d)

    pseudo_label_2d = refine_pseudo_labels(probs_2d, pseudo_label_2d, ignore_label=ignore_label) 
    pseudo_label_3d = refine_pseudo_labels(probs_3d, pseudo_label_3d, ignore_label=ignore_label) 

    closest_2d = np.argmax(cosine(features_2d, centroids_2d), axis=1)
    closest_3d = np.argmax(cosine(features_3d, centroids_3d), axis=1)

    both_valid = (pseudo_label_2d != ignore_label) & (pseudo_label_3d != ignore_label)
    one_valid = (pseudo_label_2d != ignore_label) != (pseudo_label_3d != ignore_label)
    pl_agree = pseudo_label_2d == pseudo_label_3d
    cent_agree = closest_2d == closest_3d

    pseudo_label = ignore_label * np.ones(pseudo_label_2d.shape, dtype=pseudo_label_2d.dtype)

    pseudo_label[both_valid & pl_agree] = pseudo_label_2d[both_valid & pl_agree]
    pseudo_label[both_valid & cent_agree & (pseudo_label_2d == closest_2d)] = pseudo_label_2d[both_valid & cent_agree & (pseudo_label_2d == closest_2d)]
    pseudo_label[both_valid & cent_agree & (pseudo_label_3d == closest_3d)] = pseudo_label_3d[both_valid & cent_agree & (pseudo_label_3d == closest_3d)]

    pseudo_label[~both_valid & ~one_valid & cent_agree] = closest_2d[~both_valid & ~one_valid & cent_agree]
    pseudo_label[~both_valid & one_valid & cent_agree & (pseudo_label_2d == closest_2d)] = pseudo_label_2d[~both_valid & one_valid & cent_agree & (pseudo_label_2d == closest_2d)]
    pseudo_label[~both_valid & one_valid & cent_agree & (pseudo_label_3d == closest_3d)] = pseudo_label_3d[~both_valid & one_valid & cent_agree & (pseudo_label_3d == closest_3d)]

    return pseudo_label

def agreement_filter_annoy(probs_2d=None, pseudo_label_2d=None, features_2d=None, probs_3d=None, pseudo_label_3d=None, features_3d=None, ignore_label=-100, k=100, **kwargs):
    """
    Reference: https://github.com/liyunsheng13/BDL/blob/master/SSL.py
    Per class, set the less confident half of labels to ignore label.
    :param probs: maximum probabilities (N,), where N is the number of 3D points
    :param pseudo_label: predicted label which had maximum probability (N,)
    :param ignore_label:
    :return:
    """
    pseudo_label_2d = refine_pseudo_labels(probs_2d, pseudo_label_2d, ignore_label=ignore_label) 
    pseudo_label_3d = refine_pseudo_labels(probs_3d, pseudo_label_3d, ignore_label=ignore_label) 
    
    agreement = pseudo_label_2d == pseudo_label_3d
    nonagree_idx = np.nonzero(1-agreement)[0]

    annoy_2d = AnnoyIndex(features_2d.shape[1], 'dot')
    annoy_2d.load(os.path.join(kwargs['pselab_path'], 'annoy_2d.ann'))

    consensus_2d = np.zeros(pseudo_label_2d.shape, dtype=pseudo_label_2d.dtype)
    for i, idx in enumerate(nonagree_idx):
        knn_pl = pseudo_label_2d[annoy_2d.get_nns_by_item(idx, k)]
        unique, counts = np.unique(knn_pl, return_counts=True)
        if np.max(counts) > k*0.9:
            consensus_2d[i] = unique[np.argmax(counts)]
        else:
            consensus_2d[i] = ignore_label
    del annoy_2d
    
    annoy_3d = AnnoyIndex(features_3d.shape[1], 'dot')
    annoy_3d.load(os.path.join(kwargs['pselab_path'], 'annoy_3d.ann'))
    
    consensus_3d = np.zeros(pseudo_label_3d.shape, dtype=pseudo_label_3d.dtype)
    for i, idx in enumerate(nonagree_idx):
        knn_pl = pseudo_label_3d[annoy_3d.get_nns_by_item(idx, k)]
        unique, counts = np.unique(knn_pl, return_counts=True)
        if np.max(counts) > k*0.9:
            consensus_3d[i] = unique[np.argmax(counts)]
        else:
            consensus_3d[i] = ignore_label
    del annoy_3d
    
    both_valid = (pseudo_label_2d != ignore_label) & (pseudo_label_3d != ignore_label)
    one_valid = (pseudo_label_2d != ignore_label) != (pseudo_label_3d != ignore_label)
    pl_agree = pseudo_label_2d == pseudo_label_3d
    knn_agree = consensus_2d == consensus_3d 

    pseudo_label = ignore_label * np.ones(pseudo_label_2d.shape, dtype=pseudo_label_2d.dtype)

    pseudo_label[both_valid & pl_agree] = pseudo_label_2d[both_valid & pl_agree]
    pseudo_label[both_valid & knn_agree & (pseudo_label_2d == consensus_2d)] = pseudo_label_2d[both_valid & knn_agree & (pseudo_label_2d == consensus_2d)]
    pseudo_label[both_valid & knn_agree & (pseudo_label_3d == consensus_3d)] = pseudo_label_3d[both_valid & knn_agree & (pseudo_label_3d == consensus_3d)]

    pseudo_label[~both_valid & ~one_valid & knn_agree] = consensus_2d[~both_valid & ~one_valid & knn_agree]
    pseudo_label[~both_valid & one_valid & knn_agree & (pseudo_label_2d == consensus_2d)] = pseudo_label_2d[~both_valid & one_valid & knn_agree & (pseudo_label_2d == consensus_2d)]
    pseudo_label[~both_valid & one_valid & knn_agree & (pseudo_label_3d == consensus_3d)] = pseudo_label_3d[~both_valid & one_valid & knn_agree & (pseudo_label_3d == consensus_3d)]

    return pseudo_label

def agreement_filter_statistical_test(probs_2d=None,
                                       probs_3d=None,
                                       pseudo_label_2d=None,
                                       pseudo_label_3d=None,
                                       features_2d=None,
                                       features_3d=None,
                                       ignore_label=-100,
                                       **kwargs):

    filtered_2d = certainty_filter_pl(probs_2d, ignore_label=ignore_label)
    filtered_3d = certainty_filter_pl(probs_3d, ignore_label=ignore_label)

    agree_idx = np.nonzero((filtered_2d == filtered_3d) & (filtered_2d != ignore_label))[0]
    mean_2d, var_2d = compute_moments(features_2d[agree_idx], filtered_2d[agree_idx])
    mean_3d, var_3d = compute_moments(features_3d[agree_idx], filtered_3d[agree_idx])

    logprob_2d = normal_logprob(features_2d, mean_2d, var_2d)
    logprob_3d = normal_logprob(features_3d, mean_3d, var_3d)

    N = pseudo_label_2d.shape[0]
    reject_2d = logprob_2d[np.arange(N), pseudo_label_2d] < logprob_2d[np.arange(N), pseudo_label_3d]
    reject_3d = logprob_3d[np.arange(N), pseudo_label_3d] < logprob_3d[np.arange(N), pseudo_label_2d]

    pseudo_label = ignore_label*np.ones(pseudo_label_2d.shape, dtype=pseudo_label_2d.dtype)
    pseudo_label[agree_idx] = pseudo_label_2d[agree_idx]

    recover = (filtered_2d != ignore_label) & (filtered_2d != ignore_label)
    pseudo_label[recover & ~reject_2d & reject_3d]=pseudo_label_2d[recover & ~reject_2d & reject_3d] 
    pseudo_label[recover & reject_2d & ~reject_3d]=pseudo_label_3d[recover & reject_2d & ~reject_3d] 
    
    return pseudo_label

def compute_moments(features, cls):
    c = np.max(cls) + 1
    N = features.shape[0]

    onehot = np.zeros((N, c))
    onehot[np.arange(N), cls] = 1

    mean = (onehot.T @ features)/np.sum(onehot, axis=0, keepdims=True).T
    var = (onehot.T @ (features - (onehot @ mean)) ** 2) / (np.sum(onehot, axis=0, keepdims=True) - 1).T

    return mean, var

def normal_logprob(features, mean, var):
    var = np.copy(var)
    var[var == 0.] = 1.

    power = -0.5 * np.sum((np.expand_dims(features, axis=2) - np.expand_dims(mean.T, axis=0)) ** 2 / var.T, axis=1)
    z = -0.5 * (np.sum(np.log(var), axis=1, keepdims=True) + features.shape[1] * np.log(2 * np.pi))

    return power + z.T

def soft_centroids(probs, features):
    centroids = (probs.T @ features)/np.sum(probs, axis=0, keepdims=True).T
    return centroids

def hard_centroids(pseudo_labels, features):
    max_class = int(np.max(np.unique(pseudo_labels))) + 1
    onehot = np.zeros((pseudo_labels.shape[0], max_class), dtype=features.dtype)
    onehot[np.arange(pseudo_labels.shape[0]), pseudo_labels] = 1
    return soft_centroids(onehot, features)

def cosine(A, B):
    A = A/np.sqrt(np.sum(A**2, axis=1, keepdims=True))
    B = B/np.sqrt(np.sum(B**2, axis=1, keepdims=True))

    return A @ B.T

def knn(query_idx, features, k):
    features = features/np.sqrt(np.sum(features**2, axis=1, keepdims=True))
    query = features[query_idx]

    knn = np.zeros((query.shape[0], k), dtype=np.int)
    for i in range(0, query.shape[0], 10):
        knn[i:min(i+10, query.shape[0])] = np.argpartition(query[i:min(i+10, query.shape[0])] @ features.T, -(k+1))[:, -(k+1):][:, ::-1]

    if query.shape[0] % 10:
        knn[-(query.shape[0] % 10):] = np.argpartition(query[-(query.shape[0] % 10):] @ features.T, -(k+1))[:, -(k+1):][:, ::-1]

    return knn

def knn_subsample(query_idx, features, k, N):
    features = features/np.sqrt(np.sum(features**2, axis=1, keepdims=True))
    query = features[query_idx]

    knn = np.zeros((query.shape[0], k), dtype=np.int)
    for i in range(0, query.shape[0], 10):
        knn[i:min(i+10, query.shape[0])] = np.argpartition(query[i:min(i+10, query.shape[0])] @ features.T, -(k+1))[:, -(k+1):][:, ::-1]

    if query.shape[0] % 10:
        knn[-(query.shape[0] % 10):] = np.argpartition(query[-(query.shape[0] % 10):] @ features.T, -(k+1))[:, -(k+1):][:, ::-1]

    return knn
