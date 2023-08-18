from yacs.config import CfgNode as CN

from xmuda.data.nuscenes.nuscenes_dataloader import NuScenesSCN
from xmuda.data.nuscenes_lidarseg.nuscenes_lidarseg_dataloader import NuScenesLidarSegSCN
from xmuda.data.a2d2.a2d2_dataloader import A2D2SCN
from xmuda.data.semantic_kitti.semantic_kitti_dataloader import SemanticKITTISCN

def main():
    nuscenes_usa_train = NuScenesSCN(('train_usa',),
                                     preprocess_dir = '/data/AmitRoyChowdhury/xmuda/nuscenes/preprocess',
                                     merge_classes=True)
    print('Train USA', len(nuscenes_usa_train))
    del nuscenes_usa_train
    nuscenes_usa_test = NuScenesSCN(('test_usa',),
                                     preprocess_dir = '/data/AmitRoyChowdhury/xmuda/nuscenes/preprocess',
                                     merge_classes=True)
    print('Test USA', len(nuscenes_usa_test))
    del nuscenes_usa_test
    nuscenes_singapore_train = NuScenesSCN(('train_singapore',),
                                     preprocess_dir = '/data/AmitRoyChowdhury/xmuda/nuscenes/preprocess',
                                     merge_classes=True)
    print('Train Singapore', len(nuscenes_singapore_train))
    del nuscenes_singapore_train
    nuscenes_singapore_val = NuScenesSCN(('val_singapore',),
                                     preprocess_dir = '/data/AmitRoyChowdhury/xmuda/nuscenes/preprocess',
                                     merge_classes=True)
    print('Val Singapore', len(nuscenes_singapore_val))
    del nuscenes_singapore_val
    nuscenes_singapore_test = NuScenesSCN(('test_singapore',),
                                     preprocess_dir = '/data/AmitRoyChowdhury/xmuda/nuscenes/preprocess',
                                     merge_classes=True)
    print('Test Singapore', len(nuscenes_singapore_test))
    del nuscenes_singapore_test


    nuscenes_day_train = NuScenesSCN(('train_day',),
                                     preprocess_dir = '/data/AmitRoyChowdhury/xmuda/nuscenes/preprocess',
                                     merge_classes=True)
    print('Train Day', len(nuscenes_day_train))
    del nuscenes_day_train
    nuscenes_day_test = NuScenesSCN(('test_day',),
                                     preprocess_dir = '/data/AmitRoyChowdhury/xmuda/nuscenes/preprocess',
                                     merge_classes=True)
    print('Test Day', len(nuscenes_day_test))
    del nuscenes_day_test
    nuscenes_night_train = NuScenesSCN(('train_night',),
                                     preprocess_dir = '/data/AmitRoyChowdhury/xmuda/nuscenes/preprocess',
                                     merge_classes=True)
    print('Train Night', len(nuscenes_night_train))
    del nuscenes_night_train
    nuscenes_night_val = NuScenesSCN(('val_night',),
                                     preprocess_dir = '/data/AmitRoyChowdhury/xmuda/nuscenes/preprocess',
                                     merge_classes=True)
    print('Val Night', len(nuscenes_night_val))
    del nuscenes_night_val
    nuscenes_night_test = NuScenesSCN(('test_night',),
                                     preprocess_dir = '/data/AmitRoyChowdhury/xmuda/nuscenes/preprocess',
                                     merge_classes=True)
    print('Test Night', len(nuscenes_night_test))
    del nuscenes_night_test
    
    semantic_kitti_train = SemanticKITTISCN(('train',),
                                     preprocess_dir = '/data/AmitRoyChowdhury/xmuda/SemanticKITTI/preprocess',
                                     merge_classes=True)
    print('Train', len(semantic_kitti_train))
    del semantic_kitti_train
    semantic_kitti_val = SemanticKITTISCN(('val',),
                                     preprocess_dir = '/data/AmitRoyChowdhury/xmuda/SemanticKITTI/preprocess',
                                     merge_classes=True)
    print('Val', len(semantic_kitti_val))
    del semantic_kitti_val
    semantic_kitti_test = SemanticKITTISCN(('test',),
                                     preprocess_dir = '/data/AmitRoyChowdhury/xmuda/SemanticKITTI/preprocess',
                                     merge_classes=True)
    print('Test', len(semantic_kitti_test))
    del semantic_kitti_test

if __name__ == '__main__':
    main()
