from xmuda.data.nuscenes.nuscenes_dataloader import NuScenesSCN
from xmuda.data.a2d2.a2d2_dataloader import A2D2SCN
from xmuda.data.semantic_kitti.semantic_kitti_dataloader import SemanticKITTISCN

import numpy as np

def calcStats(dataset):
    stats = {}
    stats['length'] = len(dataset)
    stats['img_sizes'] = {}
    stats['num_points'] = {}
    stats['total_points'] = 0
    stats['class_dist'] = np.zeros((len(dataset.categories),))

    for i, sample in enumerate(dataset):
        im_sz = sample['img'].shape
        if im_sz in stats['img_sizes']:
            stats['img_sizes'][im_sz] += 1
        else:
            stats['img_sizes'][im_sz] = 1

        num_points = sample['coords'].shape[0]
        if num_points in stats['num_points']:
            stats['num_points'][num_points] += 1
        else:
            stats['num_points'][num_points] = 1
        stats['total_points'] += num_points

        val, cnt = np.unique(sample['seg_label'], return_counts=True)
        for v, c in zip(val, cnt):
            if v != -100:
                stats['class_dist'][v] += c

        print('{}/{}'.format(i, stats['length']), end='\r')

    print('{}/{}'.format(stats['length'], stats['length']))
    stats['class_dist'] = stats['class_dist']/np.sum(stats['class_dist'])
    return stats

def printStats(stats):
    print('Length: {}'.format(stats['length']))
    print('Image Sizes:')
    for sz in stats['img_sizes']:
        print('\t{}: {}'.format(sz, stats['img_sizes'][sz]))
    
    print('Total Points: {}'.format(stats['total_points']))
    print('Average Points: {}'.format(stats['total_points']/float(stats['length'])))
    print('Class Counts: {}'.format(stats['class_dist']))

#dataset = NuScenesSCN(split=('train_day',),
#                      preprocess_dir='/data/AmitRoyChowdhury/xmuda/nuscenes/preprocess', 
#                      nuscenes_dir='/data/AmitRoyChowdhury/nuscenes',
#                      use_image=True,
#                      resize=None,
#                      merge_classes=True)
#printStats(calcStats(dataset))
#dataset = NuScenesSCN(split=('test_day',),
#                      preprocess_dir='/data/AmitRoyChowdhury/xmuda/nuscenes/preprocess', 
#                      nuscenes_dir='/data/AmitRoyChowdhury/nuscenes',
#                      use_image=True,
#                      resize=None,
#                      merge_classes=True)
#printStats(calcStats(dataset))
#dataset = NuScenesSCN(split=('train_night',),
#                      preprocess_dir='/data/AmitRoyChowdhury/xmuda/nuscenes/preprocess', 
#                      nuscenes_dir='/data/AmitRoyChowdhury/nuscenes',
#                      use_image=True,
#                      resize=None,
#                      merge_classes=True)
#printStats(calcStats(dataset))
#dataset = NuScenesSCN(split=('test_night',),
#                      preprocess_dir='/data/AmitRoyChowdhury/xmuda/nuscenes/preprocess', 
#                      nuscenes_dir='/data/AmitRoyChowdhury/nuscenes',
#                      use_image=True,
#                      resize=None,
#                      merge_classes=True)
#printStats(calcStats(dataset))
#dataset = NuScenesSCN(split=('val_night',),
#                      preprocess_dir='/data/AmitRoyChowdhury/xmuda/nuscenes/preprocess', 
#                      nuscenes_dir='/data/AmitRoyChowdhury/nuscenes',
#                      use_image=True,
#                      resize=None,
#                      merge_classes=True)
#printStats(calcStats(dataset))
dataset = A2D2SCN(split=('train',),
                  preprocess_dir='/data/AmitRoyChowdhury/xmuda/a2d2', 
                  use_image=True,
                  resize=None,
                  merge_classes=True)
printStats(calcStats(dataset))
dataset = A2D2SCN(split=('test',),
                  preprocess_dir='/data/AmitRoyChowdhury/xmuda/a2d2', 
                  use_image=True,
                  resize=None,
                  merge_classes=True)
printStats(calcStats(dataset))
dataset = SemanticKITTISCN(split=('train',),
                           preprocess_dir="/data/AmitRoyChowdhury/xmuda/SemanticKITTI/preprocess",
                           semantic_kitti_dir="/data/AmitRoyChowdhury/SemanticKITTI",
                           merge_classes=True)
printStats(calcStats(dataset))
dataset = SemanticKITTISCN(split=('test',),
                           preprocess_dir="/data/AmitRoyChowdhury/xmuda/SemanticKITTI/preprocess",
                           semantic_kitti_dir="/data/AmitRoyChowdhury/SemanticKITTI",
                           merge_classes=True)
printStats(calcStats(dataset))
dataset = SemanticKITTISCN(split=('val',),
                           preprocess_dir="/data/AmitRoyChowdhury/xmuda/SemanticKITTI/preprocess",
                           semantic_kitti_dir="/data/AmitRoyChowdhury/SemanticKITTI",
                           merge_classes=True)
printStats(calcStats(dataset))
