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

def entropy_combination(probs_2d, probs_3d, temperature, ignore_label=-100):
    w_2d = np.exp(-np.sum(probs_2d * np.log(probs_2d), axis=1, keepdims=True)/temperature)
    w_3d = np.exp(-np.sum(probs_3d * np.log(probs_3d), axis=1, keepdims=True)/temperature)

    probs = (w_2d * probs_2d + w_3d * probs_3d) / (w_2d + w_3d)
    pl = np.argmax(probs, axis=1)
    pl = refine_pseudo_labels(probs, pl, ignore_label=ignore_label)
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

def entropy_combination_statistical_test(probs_2d=None,
                                         probs_3d=None,
                                         pseudo_label_2d=None,
                                         pseudo_label_3d=None,
                                         features_2d=None,
                                         features_3d=None,
                                         thresholds=None,
                                         prior=None,
                                         temperature=1,
                                         ignore_label=-100,
                                         **kwargs):
    w_2d = np.exp(-np.sum(probs_2d * np.log(probs_2d), axis=1, keepdims=True)/temperature)
    w_3d = np.exp(-np.sum(probs_3d * np.log(probs_3d), axis=1, keepdims=True)/temperature)

    probs = (w_2d * probs_2d + w_3d * probs_3d) / (w_2d + w_3d)
    pl = np.argmax(probs, axis=1)

    certain_pl = refine_pseudo_labels(probs, pl, ignore_label=ignore_label)
    mean_2d, var_2d = compute_moments(features_2d[certain_pl != ignore_label], certain_pl[certain_pl != ignore_label])
    mean_3d, var_3d = compute_moments(features_3d[certain_pl != ignore_label], certain_pl[certain_pl != ignore_label])

    pseudo_label = np.copy(certain_pl)
    
    alternative_2d = (pl != pseudo_label_2d) & (certain_pl == ignore_label)
    reject_null = AB_hypothesis_testing(features_2d[alternative_2d], pl[alternative_2d], pseudo_label_2d[alternative_2d], mean_2d, var_2d, thresholds['k_ab_2d'])
    pseudo_label[alternative_2d][reject_null] = pseudo_label_2d[alternative_2d][reject_null]
    
    alternative_3d = (pl != pseudo_label_3d) & (certain_pl == ignore_label)
    reject_null = AB_hypothesis_testing(features_3d[alternative_3d], pl[alternative_3d], pseudo_label_3d[alternative_3d], mean_3d, var_3d, thresholds['k_ab_3d'])
    pseudo_label[alternative_3d][reject_null] = pseudo_label_3d[alternative_3d][reject_null]

    all_agree = (pl == pseudo_label_2d) & (pl == pseudo_label_3d) & (certain_pl == ignore_label)
    keep_null_2d = one_all_hypothesis_testing(features_2d[all_agree], pl[all_agree], mean_2d, var_2d, prior, thresholds['k_one_all_2d'], ignore_label=ignore_label)
    keep_null_3d = one_all_hypothesis_testing(features_3d[all_agree], pl[all_agree], mean_3d, var_3d, prior, thresholds['k_one_all_3d'], ignore_label=ignore_label)
    pseudo_label[all_agree][keep_null_2d & keep_null_3d] = pl[all_agree][keep_null_2d & keep_null_3d]

    return pseudo_label

def AB_hypothesis_testing(features, null_hypothesis, alternative_hypothesis, mean, var, thresholds):
    logprobs = normal_logprob(features, mean, var)

    null_logprob = logprobs[np.arange(null_hypothesis.shape[0]), null_hypothesis]
    alternative_logprob = logprobs[np.arange(alternative_hypothesis.shape[0]), alternative_hypothesis]

    return null_logprob <= alternative_logprob

def one_all_hypothesis_testing(features, null_hypothesis, mean, var, prior, thresholds, ignore_label=-100):
    probs = np.exp(normal_logprob(features, mean, var))

    null_prob = probs[np.arange(null_hypothesis.shape[0]), null_hypothesis]

    alternative_prob = np.sum(probs * prior, axis=1)
    alternative_prob = alternative_prob - (null_prob * prior[null_hypothesis])
    alternative_prob = alternative_prob/(np.sum(prior) - prior[null_hypothesis])

    return null_prob <= alternative_prob

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
