"""xMUDA experiments configuration"""
import os.path as osp

from xmuda.common.config.base import CN, _C

# public alias
cfg = _C
_C.VAL.METRIC = 'seg_iou'

# ---------------------------------------------------------------------------- #
# Specific train options
# ---------------------------------------------------------------------------- #
_C.TRAIN.CLASS_WEIGHTS = []

# ---------------------------------------------------------------------------- #
# xMUDA options
# ---------------------------------------------------------------------------- #
_C.TRAIN.XMUDA = CN()
_C.TRAIN.XMUDA.lambda_xm_src = 0.0
_C.TRAIN.XMUDA.lambda_xm_trg = 0.0
_C.TRAIN.XMUDA.lambda_seg = 0.0
_C.TRAIN.XMUDA.lambda_pl = 0.0
_C.TRAIN.XMUDA.lambda_minent = 0.0
_C.TRAIN.XMUDA.lambda_logcoral = 0.0

_C.TRAIN.XMUDA.lambda_ent = 0.0
_C.TRAIN.XMUDA.lambda_div = 0.0
_C.TRAIN.XMUDA.lambda_curr_ent = 0.0
_C.TRAIN.XMUDA.lambda_weight_div = 0.0
_C.TRAIN.XMUDA.pseudo_label_period = 0

_C.TRAIN.XMUDA.ckpt_2d = ''
_C.TRAIN.XMUDA.ckpt_3d = ''
_C.TRAIN.XMUDA.threhold_path = ''

_C.TRAIN.XMUDA.LABELING = CN()
_C.TRAIN.XMUDA.LABELING.noisy_label = False
_C.TRAIN.XMUDA.LABELING.inverse_label = False

# ---------------------------------------------------------------------------- #
# Datasets
# ---------------------------------------------------------------------------- #
_C.DATASET_SOURCE = CN()
_C.DATASET_SOURCE.TYPE = ''
_C.DATASET_SOURCE.TRAIN = tuple()
_C.DATASET_SOURCE.TEST = tuple()

_C.DATASET_TARGET = CN()
_C.DATASET_TARGET.TYPE = ''
_C.DATASET_TARGET.TRAIN = tuple()
_C.DATASET_TARGET.VAL = tuple()
_C.DATASET_TARGET.TEST = tuple()

# NuScenesSCN
_C.DATASET_SOURCE.NuScenesSCN = CN()
_C.DATASET_SOURCE.NuScenesSCN.preprocess_dir = ''
_C.DATASET_SOURCE.NuScenesSCN.nuscenes_dir = ''
_C.DATASET_SOURCE.NuScenesSCN.merge_classes = True
# 3D
_C.DATASET_SOURCE.NuScenesSCN.scale = 20
_C.DATASET_SOURCE.NuScenesSCN.full_scale = 4096
# 2D
_C.DATASET_SOURCE.NuScenesSCN.use_image = True
_C.DATASET_SOURCE.NuScenesSCN.resize = (400, 225)
_C.DATASET_SOURCE.NuScenesSCN.image_normalizer = ()
# 3D augmentation
_C.DATASET_SOURCE.NuScenesSCN.augmentation = CN()
_C.DATASET_SOURCE.NuScenesSCN.augmentation.noisy_rot = 0.1
_C.DATASET_SOURCE.NuScenesSCN.augmentation.flip_x = 0.5
_C.DATASET_SOURCE.NuScenesSCN.augmentation.rot_z = 6.2831  # 2 * pi
_C.DATASET_SOURCE.NuScenesSCN.augmentation.transl = True
# 2D augmentation
_C.DATASET_SOURCE.NuScenesSCN.augmentation.fliplr = 0.5
_C.DATASET_SOURCE.NuScenesSCN.augmentation.color_jitter = (0.4, 0.4, 0.4)
# copy over the same arguments to target dataset settings
_C.DATASET_TARGET.NuScenesSCN = CN(_C.DATASET_SOURCE.NuScenesSCN)
_C.DATASET_TARGET.NuScenesSCN.pselab_paths = None 

# NuScenesLidarSegSCN
_C.DATASET_SOURCE.NuScenesLidarSegSCN = CN()
_C.DATASET_SOURCE.NuScenesLidarSegSCN.preprocess_dir = ''
_C.DATASET_SOURCE.NuScenesLidarSegSCN.nuscenes_dir = ''
_C.DATASET_SOURCE.NuScenesLidarSegSCN.merge_classes = True
# 3D
_C.DATASET_SOURCE.NuScenesLidarSegSCN.scale = 20
_C.DATASET_SOURCE.NuScenesLidarSegSCN.full_scale = 4096
# 2D
_C.DATASET_SOURCE.NuScenesLidarSegSCN.use_image = True
_C.DATASET_SOURCE.NuScenesLidarSegSCN.resize = (400, 225)
_C.DATASET_SOURCE.NuScenesLidarSegSCN.image_normalizer = ()
# 3D augmentation
_C.DATASET_SOURCE.NuScenesLidarSegSCN.augmentation = CN()
_C.DATASET_SOURCE.NuScenesLidarSegSCN.augmentation.noisy_rot = 0.1
_C.DATASET_SOURCE.NuScenesLidarSegSCN.augmentation.flip_x = 0.5
_C.DATASET_SOURCE.NuScenesLidarSegSCN.augmentation.rot_z = 6.2831  # 2 * pi
_C.DATASET_SOURCE.NuScenesLidarSegSCN.augmentation.transl = True
# 2D augmentation
_C.DATASET_SOURCE.NuScenesLidarSegSCN.augmentation.fliplr = 0.5
_C.DATASET_SOURCE.NuScenesLidarSegSCN.augmentation.color_jitter = (0.4, 0.4, 0.4)
# copy over the same arguments to target dataset settings
_C.DATASET_TARGET.NuScenesLidarSegSCN = CN(_C.DATASET_SOURCE.NuScenesLidarSegSCN)
_C.DATASET_TARGET.NuScenesLidarSegSCN.pselab_paths = None 

# A2D2SCN
_C.DATASET_SOURCE.A2D2SCN = CN()
_C.DATASET_SOURCE.A2D2SCN.preprocess_dir = ''
_C.DATASET_SOURCE.A2D2SCN.merge_classes = True
# 3D
_C.DATASET_SOURCE.A2D2SCN.scale = 20
_C.DATASET_SOURCE.A2D2SCN.full_scale = 4096
# 2D
_C.DATASET_SOURCE.A2D2SCN.use_image = True
_C.DATASET_SOURCE.A2D2SCN.resize = (480, 302)
_C.DATASET_SOURCE.A2D2SCN.image_normalizer = ()
# 3D augmentation
_C.DATASET_SOURCE.A2D2SCN.augmentation = CN()
_C.DATASET_SOURCE.A2D2SCN.augmentation.noisy_rot = 0.1
_C.DATASET_SOURCE.A2D2SCN.augmentation.flip_y = 0.5
_C.DATASET_SOURCE.A2D2SCN.augmentation.rot_z = 6.2831  # 2 * pi
_C.DATASET_SOURCE.A2D2SCN.augmentation.transl = True
# 2D augmentation
_C.DATASET_SOURCE.A2D2SCN.augmentation.fliplr = 0.5
_C.DATASET_SOURCE.A2D2SCN.augmentation.color_jitter = (0.4, 0.4, 0.4)

# SemanticKITTISCN
_C.DATASET_SOURCE.SemanticKITTISCN = CN()
_C.DATASET_SOURCE.SemanticKITTISCN.preprocess_dir = ''
_C.DATASET_SOURCE.SemanticKITTISCN.semantic_kitti_dir = ''
_C.DATASET_SOURCE.SemanticKITTISCN.merge_classes = True
# 3D
_C.DATASET_SOURCE.SemanticKITTISCN.scale = 20
_C.DATASET_SOURCE.SemanticKITTISCN.full_scale = 4096
# 2D
_C.DATASET_SOURCE.SemanticKITTISCN.image_normalizer = ()
# 3D augmentation
_C.DATASET_SOURCE.SemanticKITTISCN.augmentation = CN()
_C.DATASET_SOURCE.SemanticKITTISCN.augmentation.noisy_rot = 0.1
_C.DATASET_SOURCE.SemanticKITTISCN.augmentation.flip_y = 0.5
_C.DATASET_SOURCE.SemanticKITTISCN.augmentation.rot_z = 6.2831  # 2 * pi
_C.DATASET_SOURCE.SemanticKITTISCN.augmentation.transl = True
# 2D augmentation
_C.DATASET_SOURCE.SemanticKITTISCN.augmentation.bottom_crop = (480, 302)
_C.DATASET_SOURCE.SemanticKITTISCN.augmentation.fliplr = 0.5
_C.DATASET_SOURCE.SemanticKITTISCN.augmentation.color_jitter = (0.4, 0.4, 0.4)
# copy over the same arguments to target dataset settings
_C.DATASET_TARGET.SemanticKITTISCN = CN(_C.DATASET_SOURCE.SemanticKITTISCN)
_C.DATASET_TARGET.SemanticKITTISCN.pselab_paths = None 

# Rellis3DSCN
_C.DATASET_SOURCE.Rellis3DSCN = CN()
_C.DATASET_SOURCE.Rellis3DSCN.preprocess_dir = ''
_C.DATASET_SOURCE.Rellis3DSCN.rellis_3d_dir = ''
_C.DATASET_SOURCE.Rellis3DSCN.merge_classes = True
# 3D
_C.DATASET_SOURCE.Rellis3DSCN.scale = 20
_C.DATASET_SOURCE.Rellis3DSCN.full_scale = 4096
# 2D
_C.DATASET_SOURCE.Rellis3DSCN.image_normalizer = ()
# 3D augmentation
_C.DATASET_SOURCE.Rellis3DSCN.augmentation = CN()
_C.DATASET_SOURCE.Rellis3DSCN.augmentation.noisy_rot = 0.1
_C.DATASET_SOURCE.Rellis3DSCN.augmentation.flip_y = 0.5
_C.DATASET_SOURCE.Rellis3DSCN.augmentation.rot_z = 6.2831  # 2 * pi
_C.DATASET_SOURCE.Rellis3DSCN.augmentation.transl = True
# 2D augmentation
_C.DATASET_SOURCE.Rellis3DSCN.augmentation.bottom_crop = (480, 302)
_C.DATASET_SOURCE.Rellis3DSCN.augmentation.fliplr = 0.5
_C.DATASET_SOURCE.Rellis3DSCN.augmentation.color_jitter = (0.4, 0.4, 0.4)
# copy over the same arguments to target dataset settings
_C.DATASET_TARGET.Rellis3DSCN = CN(_C.DATASET_SOURCE.Rellis3DSCN)
_C.DATASET_TARGET.Rellis3DSCN.pselab_paths = None 

# ---------------------------------------------------------------------------- #
# Model 2D
# ---------------------------------------------------------------------------- #
_C.MODEL_2D = CN()
_C.MODEL_2D.TYPE = ''
_C.MODEL_2D.CKPT_PATH = ''
_C.MODEL_2D.NUM_CLASSES = 5
_C.MODEL_2D.DUAL_HEAD = False
_C.MODEL_2D.FEAT_DIM = 64 
# ---------------------------------------------------------------------------- #
# UNetResNet34 options
# ---------------------------------------------------------------------------- #
_C.MODEL_2D.UNetResNet34 = CN()
_C.MODEL_2D.UNetResNet34.pretrained = True

# ---------------------------------------------------------------------------- #
# Model 3D
# ---------------------------------------------------------------------------- #
_C.MODEL_3D = CN()
_C.MODEL_3D.TYPE = ''
_C.MODEL_3D.CKPT_PATH = ''
_C.MODEL_3D.NUM_CLASSES = 5
_C.MODEL_3D.DUAL_HEAD = False
_C.MODEL_3D.FEAT_DIM = 16 
# ----------------------------------------------------------------------------- #
# SCN options
# ----------------------------------------------------------------------------- #
_C.MODEL_3D.SCN = CN()
_C.MODEL_3D.SCN.in_channels = 1
_C.MODEL_3D.SCN.m = 16  # number of unet features (multiplied in each layer)
_C.MODEL_3D.SCN.block_reps = 1  # block repetitions
_C.MODEL_3D.SCN.residual_blocks = False  # ResNet style basic blocks
_C.MODEL_3D.SCN.full_scale = 4096
_C.MODEL_3D.SCN.num_planes = 7

# ----------------------------------------------------------------------------- #
# AutoEncoder options
# ----------------------------------------------------------------------------- #
_C.AE = CN()
_C.AE.EMBED_DIM = 64 
_C.AE.HIDDEN_LAYERS = 1

# ----------------------------------------------------------------------------- #
# Genertor options
# ----------------------------------------------------------------------------- #
_C.GENERATOR = CN()
_C.GENERATOR.EMBED_DIM = 64 

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# @ will be replaced by config path
_C.OUTPUT_DIR = osp.expanduser('~/workspace/outputs/xmuda/@')
