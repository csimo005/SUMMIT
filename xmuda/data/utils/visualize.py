import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import numpy as np
from xmuda.data.utils.turbo_cmap import interpolate_or_clip, turbo_colormap_data

RELLIS_3D_ID_TO_BGR = {0: (0, 0, 0),
                       1: (20, 64, 108),
                       3: (0, 102, 0),
                       4: (0, 255, 0),
                       5: (153, 153, 0),
                       6: (255, 128, 0),
                       7: (255, 0, 0),
                       8: (0, 255, 255),
                       9: (127, 0, 255),
                       10: (64, 64, 64),
                       12: (0, 0, 255),
                       15: (0, 0, 102),
                       17: (255, 153, 204),
                       18: (204, 0, 102),
                       19: (204, 153, 255),
                       23: (170, 170, 170),
                       27: (255, 121, 41),
                       29: (255, 31, 101),
                       30: (9, 149, 137),
                       31: (239, 255, 134),
                       33: (34, 66, 99),
                       34: (138, 22, 110)}

RELLIS_3D_COLOR_PALETTE = [RELLIS_3D_ID_TO_BGR[id] if id in RELLIS_3D_ID_TO_BGR.keys() else [0, 0, 0]
                                for id in range(list(RELLIS_3D_ID_TO_BGR.keys())[-1] + 1)]

RELLIS_3D_COLOR_PALETTE_SHORT_BGR = [
    [255,   0,   0],  # sky
    [ 20,  64, 108],  # ground 
    [255,  128,  0],  # water
    [  0,  255,  0],  # vegetation
    [255, 153, 204],  # person 
    [ 64,  64,  64],  # road
    [  0,   0, 255],  # building
    [255, 121,  41],  # barrier
    [  0, 255, 255],  # vehicle 
    [255,  31, 101],  # incline
    [127,   0, 255], # other objects
    [  0,   0,   0],  # ignore
]
RELLIS_3D_COLOR_PALETTE_SHORT = [(c[2], c[1], c[0]) for c in RELLIS_3D_COLOR_PALETTE_SHORT_BGR]

# all classes
NUSCENES_COLOR_PALETTE = [
    (255, 158, 0),  # car
    (255, 158, 0),  # truck
    (255, 158, 0),  # bus
    (255, 158, 0),  # trailer
    (255, 158, 0),  # construction_vehicle
    (0, 0, 230),  # pedestrian
    (255, 61, 99),  # motorcycle
    (255, 61, 99),  # bicycle
    (0, 0, 0),  # traffic_cone
    (0, 0, 0),  # barrier
    (200, 200, 200),  # background
]

# classes after merging (as used in xMUDA)
NUSCENES_COLOR_PALETTE_SHORT = [
    (255, 158, 0),  # vehicle
    (0, 0, 230),  # pedestrian
    (255, 61, 99),  # bike
    (0, 0, 0),  # traffic boundary
    (200, 200, 200),  # background
]

# all classes
A2D2_COLOR_PALETTE_SHORT = [
    (255, 0, 0),  # car
    (255, 128, 0),  # truck
    (182, 89, 6),  # bike
    (204, 153, 255),  # person
    (255, 0, 255),  # road
    (150, 150, 200),  # parking
    (180, 150, 200),  # sidewalk
    (241, 230, 255),  # building
    (147, 253, 194),  # nature
    (255, 246, 143),  # other-objects
    (0, 0, 0)  # ignore
]

# colors as defined in https://github.com/PRBonn/semantic-kitti-api/blob/master/config/semantic-kitti.yaml
SEMANTIC_KITTI_ID_TO_BGR = {  # bgr
  0: [0, 0, 0],
  1: [0, 0, 255],
  10: [245, 150, 100],
  11: [245, 230, 100],
  13: [250, 80, 100],
  15: [150, 60, 30],
  16: [255, 0, 0],
  18: [180, 30, 80],
  20: [255, 0, 0],
  30: [30, 30, 255],
  31: [200, 40, 255],
  32: [90, 30, 150],
  40: [255, 0, 255],
  44: [255, 150, 255],
  48: [75, 0, 75],
  49: [75, 0, 175],
  50: [0, 200, 255],
  51: [50, 120, 255],
  52: [0, 150, 255],
  60: [170, 255, 150],
  70: [0, 175, 0],
  71: [0, 60, 135],
  72: [80, 240, 150],
  80: [150, 240, 255],
  81: [0, 0, 255],
  99: [255, 255, 50],
  252: [245, 150, 100],
  256: [255, 0, 0],
  253: [200, 40, 255],
  254: [30, 30, 255],
  255: [90, 30, 150],
  257: [250, 80, 100],
  258: [180, 30, 80],
  259: [255, 0, 0],
}
SEMANTIC_KITTI_COLOR_PALETTE = [SEMANTIC_KITTI_ID_TO_BGR[id] if id in SEMANTIC_KITTI_ID_TO_BGR.keys() else [0, 0, 0]
                                for id in range(list(SEMANTIC_KITTI_ID_TO_BGR.keys())[-1] + 1)]


# classes after merging (as used in xMUDA)
SEMANTIC_KITTI_COLOR_PALETTE_SHORT_BGR = [
    [245, 150, 100],  # car
    [180, 30, 80],  # truck
    [150, 60, 30],  # bike
    [30, 30, 255],  # person
    [255, 0, 255],  # road
    [255, 150, 255],  # parking
    [75, 0, 75],  # sidewalk
    [0, 200, 255],  # building
    [0, 175, 0],  # nature
    [255, 255, 50],  # other-objects
    [0, 0, 0],  # ignore
]
SEMANTIC_KITTI_COLOR_PALETTE_SHORT = [(c[2], c[1], c[0]) for c in SEMANTIC_KITTI_COLOR_PALETTE_SHORT_BGR]


def draw_points_image_labels(img, img_indices, seg_labels, save_pth=None, color_palette_type='NuScenes', point_size=0.5):
    if color_palette_type == 'NuScenes':
        color_palette = NUSCENES_COLOR_PALETTE_SHORT
    elif color_palette_type == 'A2D2':
        color_palette = A2D2_COLOR_PALETTE_SHORT
    elif color_palette_type == 'SemanticKITTI':
        color_palette = SEMANTIC_KITTI_COLOR_PALETTE_SHORT
    elif color_palette_type == 'SemanticKITTI_long':
        color_palette = SEMANTIC_KITTI_COLOR_PALETTE
    elif color_palette_type == 'Rellis3D':
        color_palette = RELLIS_3D_COLOR_PALETTE_SHORT
    elif color_palette_type == 'Rellis3D_long':
        color_palette = RELLIS_3D_COLOR_PALETTE
    else:
        raise NotImplementedError('Color palette type not supported')
    color_palette = np.array(color_palette) / 255.
    seg_labels[seg_labels == -100] = len(color_palette) - 1
    colors = color_palette[seg_labels]

    plt.clf()
    if img is not None:
        plt.imshow(img)
    plt.scatter(img_indices[:, 1], img_indices[:, 0], c=colors, alpha=0.5, s=point_size)

    plt.axis('off')

    if save_pth:
        plt.savefig(save_pth)
    else:
        plt.show()

def draw_points_pred_gt(img, img_indices, pred, seg_labels, save_pth=None, color_palette_type='NuScenes', point_size=0.5):
    if color_palette_type == 'NuScenes':
        color_palette = NUSCENES_COLOR_PALETTE_SHORT
    elif color_palette_type == 'A2D2':
        color_palette = A2D2_COLOR_PALETTE_SHORT
    elif color_palette_type == 'SemanticKITTI':
        color_palette = SEMANTIC_KITTI_COLOR_PALETTE_SHORT
    elif color_palette_type == 'SemanticKITTI_long':
        color_palette = SEMANTIC_KITTI_COLOR_PALETTE
    elif color_palette_type == 'Rellis3D':
        color_palette = RELLIS_3D_COLOR_PALETTE_SHORT
    elif color_palette_type == 'Rellis3D_long':
        color_palette = RELLIS_3D_COLOR_PALETTE
    else:
        raise NotImplementedError('Color palette type not supported')
    color_palette = np.array(color_palette) / 255.
    seg_labels[seg_labels == -100] = len(color_palette) - 1
    colors = color_palette[seg_labels]

    edge_palette = np.asarray([[1., 0., 0.], [0., 1., 0.]])
    colors = edge_palette[1 * (pred == seg_labels)]

    plt.clf()
    if img is not None:
        plt.imshow(img)
    x = (img_indices[:, 1]-np.min(img_indices[:, 1]))/(np.max(img_indices[:, 1]) - np.min(img_indices[:, 1]))
    y = (img_indices[:, 0]-np.min(img_indices[:, 0]))/(np.max(img_indices[:, 0]) - np.min(img_indices[:, 0]))
    plt.scatter(x*img.shape[1], y*img.shape[0], c=colors, alpha=0.5, s=point_size)

    plt.axis('off')

    if save_pth:
        plt.savefig(save_pth)
    else:
        plt.show()

def draw_points_pred_agree(img, img_indices, pred_2d, pred_3d, save_pth=None, color_palette_type='NuScenes', point_size=0.5):
    if color_palette_type == 'NuScenes':
        color_palette = NUSCENES_COLOR_PALETTE_SHORT
    elif color_palette_type == 'A2D2':
        color_palette = A2D2_COLOR_PALETTE_SHORT
    elif color_palette_type == 'SemanticKITTI':
        color_palette = SEMANTIC_KITTI_COLOR_PALETTE_SHORT
    elif color_palette_type == 'SemanticKITTI_long':
        color_palette = SEMANTIC_KITTI_COLOR_PALETTE
    elif color_palette_type == 'Rellis3D':
        color_palette = RELLIS_3D_COLOR_PALETTE_SHORT
    elif color_palette_type == 'Rellis3D_long':
        color_palette = RELLIS_3D_COLOR_PALETTE
    else:
        raise NotImplementedError('Color palette type not supported')
    palette = np.asarray([[1., 0., 0.], [0., 1., 0.]])
    colors = palette[1 * (pred_2d == pred_3d)]

    plt.clf()
    if img is not None:
        plt.imshow(img)
    x = (img_indices[:, 1]-np.min(img_indices[:, 1]))/(np.max(img_indices[:, 1]) - np.min(img_indices[:, 1]))
    y = (img_indices[:, 0]-np.min(img_indices[:, 0]))/(np.max(img_indices[:, 0]) - np.min(img_indices[:, 0]))
    plt.scatter(x*img.shape[1], y*img.shape[0], c=colors, alpha=0.5, s=point_size)

    plt.axis('off')

    if save_pth:
        plt.savefig(save_pth)
    else:
        plt.show()

def normalize_depth(depth, d_min, d_max):
    # normalize linearly between d_min and d_max
    data = np.clip(depth, d_min, d_max)
    return (data - d_min) / (d_max - d_min)


def draw_points_image_depth(img, img_indices, depth, save_pth=None, point_size=0.5):
    # depth = normalize_depth(depth, d_min=3., d_max=50.)
    depth = normalize_depth(depth, d_min=depth.min(), d_max=depth.max())
    colors = []
    for depth_val in depth:
        colors.append(interpolate_or_clip(colormap=turbo_colormap_data, x=depth_val))
    # ax5.imshow(np.full_like(img, 255))
    plt.clf()
    plt.imshow(img)
    plt.scatter(img_indices[:, 1], img_indices[:, 0], c=colors, alpha=0.5, s=point_size)

    plt.axis('off')

    if save_pth:
        plt.savefig(save_pth)
    else:
        plt.show()


def draw_bird_eye_view(coords, full_scale=4096):
    plt.scatter(coords[:, 0], coords[:, 1], s=0.1)
    plt.xlim([0, full_scale])
    plt.ylim([0, full_scale])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

def draw_seg_image(img, save_pth=None, color_palette_type='NuScenes'):
    if color_palette_type == 'NuScenes':
        color_palette = NUSCENES_COLOR_PALETTE_SHORT
    elif color_palette_type == 'A2D2':
        color_palette = A2D2_COLOR_PALETTE_SHORT
    elif color_palette_type == 'SemanticKITTI':
        color_palette = SEMANTIC_KITTI_COLOR_PALETTE_SHORT
    elif color_palette_type == 'SemanticKITTI_long':
        color_palette = SEMANTIC_KITTI_COLOR_PALETTE
    elif color_palette_type == 'Rellis3D':
        color_palette = RELLIS_3D_COLOR_PALETTE_SHORT
    elif color_palette_type == 'Rellis3D_long':
        color_palette = RELLIS_3D_COLOR_PALETTE
    else:
        raise NotImplementedError('Color palette type not supported')
    color_palette = np.array(color_palette) / 255.
    img[img == -100] = len(color_palette) - 1

    img = np.reshape(color_palette[img.flatten()], (*img.shape, 3))

    plt.clf()
    plt.imshow(img)

    if save_pth:
        plt.savefig(save_pth)
    else:
        plt.show()

class VideoWriter():
    def __init__(self, output_path, visualize_func, dataset_type):
        self.output_path = output_path
        self.visualize_func = visualize_func
        self.dataset_type = dataset_type

        self._fig = None 
        self._ax = None
        self._output_path = None

    def initVideo(self, output_path):
        return

    def update(self, rgb_path, draw_args, draw_kwargs):
        return
        
    def write(self):
        return

    def A2D2_VideoPath(self, rgb_path):
        return

    def NuScenes_VideoPath(self, rgb_path):
        return
    
    def SemanticKITTI_VideoPath(self, rgb_path):
        fields = rgb_path.split('/')
        return ''
        return
