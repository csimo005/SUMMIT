import numpy as np
import numpy.random as nr
from scipy.optimize import fsolve

import time
import argparse
import os

def certainty_filter(probs, medians, ignore_label=-100):
    pl = np.argmax(probs, axis=1)
    probs = probs[np.arange(pl.shape[0]), pl]

    for cls in np.unique(pl):
        cls_idx = np.nonzero(pl == cls)[0]

        thresh = min(medians[cls], 0.9)
        ignore_idx = cls_idx[probs[cls_idx] < thresh]
        pl[ignore_idx] = ignore_label

    return pl

def entropy_weight(probs_2d, probs_3d):
    probs_2d[probs_2d < 1e-10] = 1.
    w_2d = np.exp(np.sum(probs_2d * np.log(probs_2d), axis=1, keepdims=True))

    probs_3d[probs_3d < 1e-10] = 1.
    w_3d = np.exp(np.sum(probs_3d * np.log(probs_3d), axis=1, keepdims=True))

    probs = (w_2d * probs_2d + w_3d * probs_3d)/(w_2d + w_3d)
    return probs

def agreement_filter(pseudo_label_2d, pseudo_label_3d, ignore_label=-100):
    agreed = pseudo_label_2d == pseudo_label_3d
    pl = ignore_label * np.ones(pseudo_label_2d.shape, dtype=pseudo_label_2d.dtype)
    pl[agreed] = pseudo_label_2d[agreed]

    return pl

def compute_moments(files, cls, key, ignore_label=-100):
    print('Computing moments for {}'.format(key))
    C = np.max(cls) + 1
    F = np.load(files[0], allow_pickle=True)[key].shape[1]
    z = np.zeros((1, C))

    itr = 0
    mean = np.zeros((C, F))
    for i, fname in enumerate(files): 
        print('mean {}/{}'.format(i+1, len(files)))
        features = np.load(fname, allow_pickle=True)[key]
        N = features.shape[0]
 
        onehot = np.zeros((N, C))
        onehot[np.arange(N)[cls[itr: itr+N] != ignore_label], cls[itr: itr+N][cls[itr: itr+N] != ignore_label]] = 1
        mean += onehot.T @ features
        z += np.sum(onehot, axis=0, keepdims=True)

        itr += N

    z[z == 0] = 1
    mean = mean / z.T

    itr = 0
    var = np.zeros((C, F)) 
    for i, fname in enumerate(files): 
        print('var {}/{}'.format(i+1, len(files)))
        features = np.load(fname, allow_pickle=True)[key]
        N = features.shape[0]
 
        onehot = np.zeros((N, C))
        onehot[np.arange(N)[cls[itr: itr+N] != ignore_label], cls[itr: itr+N][cls[itr: itr+N] != ignore_label]] = 1
        var += onehot.T @ (features - (onehot @ mean)) ** 2

        itr += N

    var = var / z.T

    return mean, var

def median_prob(files, key):
    bin_count = 1000 
    histogram = None
    for i, fname in enumerate(files):
        print('median {}/{}'.format(i+1, len(files)))
        with np.load(fname, allow_pickle=True) as data:
            probs = data[key]
            pl = np.argmax(probs, axis=1)
            if i == 0:
                c = probs.shape[1]
                histogram = np.zeros((c, bin_count,), dtype=np.ulonglong)
            for j in range(probs.shape[1]):
                cls_idx = np.nonzero(pl == j)[0]
                hist = np.histogram(probs[cls_idx, j], bins=bin_count, range=(0,1))[0]
                histogram[j] += hist.astype(np.ulonglong)

    median = np.zeros((histogram.shape[0],))
    for i in range(median.shape[0]):
        csum = np.cumsum(histogram[i])
        median[i] = np.argmin(np.abs(csum - csum[-1]/2))/bin_count
    return median

def mean_prob(files):
    mean_2d = None
    mean_3d = None
    N = 0.
    for i, fname in enumerate(files):
        print('mean {}/{}'.format(i+1, len(files)))
        with np.load(fname, allow_pickle=True) as data:
            if i == 0:
                mean_2d = np.mean(data['probs_2d'], axis=0)
                mean_3d = np.mean(data['probs_3d'], axis=0)
                N = data['probs_2d'].shape[0]
            else:
                m2d = np.mean(data['probs_2d'], axis=0) 
                m3d = np.mean(data['probs_3d'], axis=0)
                M = float(data['probs_2d'].shape[0])

                mean_2d = (N*mean_2d + M*m2d) / (N+M)
                mean_3d = (N*mean_3d + M*m3d) / (N+M)
                N = N + M
    
    return mean_2d, mean_3d

def median_entropy_prob(files):
    bin_count = 1000
    histogram = None
    for i, fname in enumerate(files):
        print('median {}/{}'.format(i+1, len(files)))
        with np.load(fname, allow_pickle=True) as data:
            probs = entropy_weight(data['probs_2d'], data['probs_3d'])
            pl = np.argmax(probs, axis=1)
            if i == 0:
                C = probs.shape[1]
                histogram = np.zeros((C, bin_count,), dtype=np.ulonglong)
            for j in range(probs.shape[1]):
                cls_idx = np.nonzero(pl == j)[0]
                hist = np.histogram(probs[cls_idx, j], bins=bin_count, range=(0,1))[0]
                histogram[j] += hist.astype(np.ulonglong)

    median = np.zeros((histogram.shape[0],))
    for i in range(median.shape[0]):
        csum = np.cumsum(histogram[i])
        median[i] = np.argmin(np.abs(csum - csum[-1]/2))/bin_count
    return median

def estimate_prior(probs_2d, probs_3d):
    prior_2d = np.mean(probs_2d, axis=0)
    prior_2d = prior_2d/np.sum(prior_2d)
    
    prior_3d = np.mean(probs_3d, axis=0)
    prior_3d = prior_3d/np.sum(prior_3d)

    return (prior_2d + prior_3d) / 2.

def normal(x, mean, var):
    var = np.copy(var)
    var[var == 0.] = 1e-6

    power = -0.5 * np.sum((np.expand_dims(x, axis=2) - np.expand_dims(mean.T, axis=0)) ** 2/ var.T, axis=1)
    z = np.sqrt(np.prod(var, axis=1, keepdims=True) * (2 * np.pi) ** x.shape[1])
    z[z == 0.] = 1e-6

    return np.exp(power)/z.T

def mahalanobis_dist(x, mean, var):
    var = np.copy(var)
    var[var == 0.] = 1e-6

    dist = np.sqrt(np.sum((np.expand_dims(x, axis=2) - np.expand_dims(mean.T, axis=0)) ** 2/ var.T, axis=1))
    return dist 

def AB_hypothesis_test(features, null_hypothesis, altr_hypothesis, mean, var, k):
    probs = normal(features, mean, var)

    N = null_hypothesis.shape[0]
    null_prob = probs[np.arange(N), null_hypothesis]
    altr_prob = probs[np.arange(N), altr_hypothesis]

    return null_prob <= k * altr_prob

def one_all_hypothesis_test(features, null_hypothesis, altr_hypothesis, mean, var, prior, k):
    probs = normal(features, mean, var)

    null_prob = probs[null_hypothesis]
    altr_prob = np.sum(prior*probs, axis=1) - prior[null_hypothesis] * null_prob
    altr_prob = altr_prob/(np.sum(prior) - prior[null_hypothesis])

    return null_prob <= k * altr_prob

def mahalanobis_test(features, null_hypothesis, altr_hypothesis, mean, var, k):
    dist = mahalanobis_dist(features, mean, var)

    N = null_hypothesis.shape[0]
    null_dist = dist[np.arange(N), null_hypothesis]
    altr_dist = dist[np.arange(N), altr_hypothesis]

    return null_dist >= altr_dist

def main(args):
    files = os.listdir(args.features_dir)
    files.sort()
    files = [os.path.join(args.features_dir, fname) for fname in files]

    if args.filter == 'EW':
        print('Entropy Weight')
        median_entropy = median_entropy_prob(files)
        refined = []
        for fname in files:
            data = np.load(fname, allow_pickle=True)
            refined += [certainty_filter(entropy_weight(data['probs_2d'], data['probs_3d']), median_entropy)]
        refined_pseudo_label = np.concatenate(refined, axis=0)
        if args.HT:
            mean_2d, var_2d = compute_moments(files, refined_pseudo_label, 'features_2d')
            mean_3d, var_3d = compute_moments(files, refined_pseudo_label, 'features_3d')

            itr = 0
            for fname in files:
                data = np.load(fname, allow_pickle=True)
                probs_2d = data['probs_2d']
                probs_3d = data['probs_3d']
                features_2d = data['features_2d']
                features_3d = data['features_3d']
                N = probs_2d.shape[0]

                unfiltered_2d = np.argmax(probs_2d, axis=1)
                unfiltered_3d = np.argmax(probs_3d, axis=1)
                
                reject_2d = np.zeros(unfiltered_2d.shape, dtype=bool)
                reject_2d[:N//2] = AB_hypothesis_test(features_2d[:N//2],
                                                      unfiltered_2d[:N//2],
                                                      unfiltered_3d[:N//2],
                                                      mean_2d, var_2d, args.k)
                reject_2d[N//2:] = AB_hypothesis_test(features_2d[N//2:],
                                                      unfiltered_2d[N//2:],
                                                      unfiltered_3d[N//2:],
                                                      mean_2d, var_2d, args.k)

                reject_3d = np.zeros(unfiltered_3d.shape, dtype=bool)
                reject_3d[:N//2] = AB_hypothesis_test(features_3d[:N//2],
                                                      unfiltered_3d[:N//2],
                                                      unfiltered_3d[:N//2],
                                                      mean_3d, var_3d, args.k)
                reject_3d[N//2:] = AB_hypothesis_test(features_3d[N//2:],
                                                      unfiltered_3d[N//2:],
                                                      unfiltered_3d[N//2:],
                                                      mean_3d, var_3d, args.k)
   
                invalid_disagree = (refined_pseudo_label[itr:itr+N] == -100) & (unfiltered_2d != unfiltered_3d)
                take_2d = invalid_disagree & ~reject_2d & reject_3d
                take_2d = np.nonzero(take_2d)[0]
                refined_pseudo_label[itr:itr+N][take_2d] = unfiltered_2d[take_2d]
    
                take_3d = invalid_disagree & reject_2d & ~reject_3d
                take_3d = np.nonzero(take_3d)[0]
                refined_pseudo_label[itr:itr+N][take_3d] = unfiltered_3d[take_3d]

                itr += N 
        np.savez(args.output_path, refined_pseudo_label=refined_pseudo_label)
    elif args.filter == 'AF':
        print('Agreement Filter')
        median_2d = median_prob(files, 'probs_2d')
        print(median_2d)
        median_3d = median_prob(files, 'probs_3d')
        print(median_3d)

        refined = []
        for fname in files:
            data = np.load(fname, allow_pickle=True)
            refined += [agreement_filter(certainty_filter(data['probs_2d'], median_2d),
                                         certainty_filter(data['probs_3d'], median_3d))]
        refined_pseudo_label = np.concatenate(refined, axis=0)
        
        if args.HT:
            print('Applying Hypothesis Testing')
            mean_2d, var_2d = compute_moments(files, refined_pseudo_label, 'features_2d')
            mean_3d, var_3d = compute_moments(files, refined_pseudo_label, 'features_3d')

            itr = 0
            for fname in files:
                data = np.load(fname, allow_pickle=True)
                probs_2d = data['probs_2d']
                probs_3d = data['probs_3d']
                features_2d = data['features_2d']
                features_3d = data['features_3d']
                N = probs_2d.shape[0]

                unfiltered_2d = np.argmax(probs_2d, axis=1)
                unfiltered_3d = np.argmax(probs_3d, axis=1)
                
                reject_2d = np.zeros(unfiltered_2d.shape, dtype=bool)
                reject_2d[:N//2] = AB_hypothesis_test(features_2d[:N//2],
                                                      unfiltered_2d[:N//2],
                                                      unfiltered_3d[:N//2],
                                                      mean_2d, var_2d, args.k)
                reject_2d[N//2:] = AB_hypothesis_test(features_2d[N//2:],
                                                      unfiltered_2d[N//2:],
                                                      unfiltered_3d[N//2:],
                                                      mean_2d, var_2d, args.k)

                reject_3d = np.zeros(unfiltered_3d.shape, dtype=bool)
                reject_3d[:N//2] = AB_hypothesis_test(features_3d[:N//2],
                                                      unfiltered_3d[:N//2],
                                                      unfiltered_3d[:N//2],
                                                      mean_3d, var_3d, args.k)
                reject_3d[N//2:] = AB_hypothesis_test(features_3d[N//2:],
                                                      unfiltered_3d[N//2:],
                                                      unfiltered_3d[N//2:],
                                                      mean_3d, var_3d, args.k)
   
                take_2d = (refined_pseudo_label[itr: itr+N] == -100) & ~reject_2d & reject_3d
                take_2d = np.nonzero(take_2d)[0]
                refined_pseudo_label[itr:itr+N][take_2d] = unfiltered_2d[take_2d]
    
                take_3d = (refined_pseudo_label[itr: itr+N] == -100) & reject_2d & ~reject_3d
                take_3d = np.nonzero(take_3d)[0]
                refined_pseudo_label[itr:itr+N][take_3d] = unfiltered_3d[take_3d]

                itr += N 
        np.savez(args.output_path, refined_pseudo_label=refined_pseudo_label)
    elif args.filter == 'MH':
        print('Entropy Weight')
        median_entropy = median_entropy_prob(files)
        refined = []
        for fname in files:
            data = np.load(fname, allow_pickle=True)
            refined += [certainty_filter(entropy_weight(data['probs_2d'], data['probs_3d']), median_entropy)]
        refined_pseudo_label = np.concatenate(refined, axis=0)
        
        mean_2d, var_2d = compute_moments(files, refined_pseudo_label, 'features_2d')
        mean_3d, var_3d = compute_moments(files, refined_pseudo_label, 'features_3d')

        itr = 0
        for fname in files:
            data = np.load(fname, allow_pickle=True)
            probs_2d = data['probs_2d']
            probs_3d = data['probs_3d']
            features_2d = data['features_2d']
            features_3d = data['features_3d']
            N = probs_2d.shape[0]

            unfiltered_2d = np.argmax(probs_2d, axis=1)
            unfiltered_3d = np.argmax(probs_3d, axis=1)
            
            reject_2d = np.zeros(unfiltered_2d.shape, dtype=bool)
            reject_2d[:N//2] = mahalanobis_test(features_2d[:N//2],
                                                unfiltered_2d[:N//2],
                                                unfiltered_3d[:N//2],
                                                mean_2d, var_2d, args.k)
            reject_2d[N//2:] = mahalanobis_test(features_2d[N//2:],
                                                unfiltered_2d[N//2:],
                                                unfiltered_3d[N//2:],
                                                mean_2d, var_2d, args.k)

            reject_3d = np.zeros(unfiltered_3d.shape, dtype=bool)
            reject_3d[:N//2] = mahalanobis_test(features_3d[:N//2],
                                                unfiltered_3d[:N//2],
                                                unfiltered_3d[:N//2],
                                                mean_3d, var_3d, args.k)
            reject_3d[N//2:] = mahalanobis_test(features_3d[N//2:],
                                                unfiltered_3d[N//2:],
                                                unfiltered_3d[N//2:],
                                                mean_3d, var_3d, args.k)
   
            invalid_disagree = (refined_pseudo_label[itr:itr+N] == -100) & (unfiltered_2d != unfiltered_3d)
            take_2d = invalid_disagree & ~reject_2d & reject_3d
            take_2d = np.nonzero(take_2d)[0]
            refined_pseudo_label[itr:itr+N][take_2d] = unfiltered_2d[take_2d]

            take_3d = invalid_disagree & reject_2d & ~reject_3d
            take_3d = np.nonzero(take_3d)[0]
            refined_pseudo_label[itr:itr+N][take_3d] = unfiltered_3d[take_3d]

            itr += N 
        np.savez(args.output_path, refined_pseudo_label=refined_pseudo_label)
        np.savez(args.output_path, refined_pseudo_label=refined_pseudo_label)
    elif args.filter == 'AF+EW':
        print('Agreement Filter + Entropy Weighting')
        median_2d, median_3d = median_prob(files)
        median_entropy = compute_entropy_medians(files)

        refined = []
        for fname in files:
            data = np.load(fname, allow_pickle=True)
            probs_2d = data['probs_2d']
            probs_3d = data['probs_3d']

            pl = agreement_filter(certainty_filter(probs_2d, median_2d),
                                  certainty_filter(probs_3d, median_3d))
            pl[pl == -100] = certainty_filter(entropy_weight(probs_2d[pl == -100],
                                                             probs_3d[pl == -100]),
                                              median_entropy)
            refined += [pl]
        refined_pseudo_label = np.concatenate(refined, axis=0)
        if args.HT:
            mean_2d, var_2d = compute_moments(files, refined_pseudo_label, 'features_2d')
            mean_3d, var_3d = compute_moments(files, refined_pseudo_label, 'features_3d')

            itr = 0
            for fname in files:
                data = np.load(fname, allow_pickle=True)
                probs_2d = data['probs_2d']
                probs_3d = data['probs_3d']
                features_2d = data['features_2d']
                features_3d = data['features_3d']

                unfiltered_2d = np.argmax(probs_2d, axis=1)
                unfiltered_3d = np.argmax(probs_3d, axis=1)
    
                reject_2d = AB_hypothesis_test(features_2d,
                                               unfiltered_2d,
                                               unfiltered_3d,
                                               mean_2d, var_2d, args.k)
                reject_3d = AB_hypothesis_test(features_3d,
                                               unfiltered_3d,
                                               unfiltered_2d,
                                               mean_3d, var_3d, args.k)
   
                N = probs_2d.shape[0] 
                take_2d = (refined_pseudo_label[itr: itr+N] == -100) & ~reject_2d & reject_3d
                take_2d = np.nonzero(take_2d)[0]
                refined_pseudo_label[itr:itr+N][take_2d] = unfiltered_2d[take_2d]
    
                take_3d = (refined_pseudo_label[itr: itr+N] == -100) & reject_2d & ~reject_3d
                take_3d = np.nonzero(take_3d)[0]
                refined_pseudo_label[itr:itr+N][take_3d] = unfiltered_3d[take_3d]

                itr += N 
        np.savez(args.output_path, refined_pseudo_label=refined_pseudo_label)
    else:
        print('Certainty Filter')
        median_2d, median_3d = median_prob(files)
        refined_2d = []
        refined_3d = []
        for fname in files:
            data = np.load(fname, allow_pickle=True)
            refined_2d += [certainty_filter(data['probs_2d'], median_2d)]
            refined_3d += [certainty_filter(data['probs_3d'], median_3d)]
        refined_pseudo_label_2d = np.concatenate(refined_2d, axis=0)
        refined_pseudo_label_3d = np.concatenate(refined_3d, axis=0)
        np.savez(args.output_path, refined_pseudo_label_2d=refined_pseudo_label_2d, refined_pseudo_label_3d=refined_pseudo_label_3d)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Precompute refined pseudo labels')
    parser.add_argument('features_dir', type=str, help='Path to directory containing features')
    parser.add_argument('output_path', type=str, help='Path to save refined pseudo labels')
    parser.add_argument('--filter', type=str, choices=['AF', 'EW', 'AF+EW', 'MH', 'None'], default='None', help='Which filter to use')
    parser.add_argument('--HT', action='store_true', help='Flag to do hypothesis testing')
    parser.add_argument('--k', type=float, default=1.0, help='Threshold for hypothesis testing')

    args = parser.parse_args()
    main(args)
