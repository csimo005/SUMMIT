import os
import os.path as osp
import numpy as np
import pickle
from pyquaternion import Quaternion

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import points_in_box
from nuscenes.eval.detection.utils import category_to_detection_name

from xmuda.data.nuscenes_lidarseg.projection import map_pointcloud_to_image
from xmuda.data.nuscenes_lidarseg import splits

def preprocess(nusc, split_names, root_dir, out_dir,
               keyword=None, keyword_action=None, subset_name=None,
               location=None):
    # cannot process day/night and location at the same time
    assert not (bool(keyword) and bool(location))
    if keyword:
        assert keyword_action in ['filter', 'exclude']

    # init dict to save
    pkl_dict = {}
    for split_name in split_names:
         pkl_dict[split_name] = []

    for i, sample in enumerate(nusc.sample):
        curr_scene_name = nusc.get('scene', sample['scene_token'])['name']

        # get if the current scene is in train, val or test
        curr_split = None
        for split_name in split_names:
            if curr_scene_name in getattr(splits, split_name):
                curr_split = split_name
                break
        if curr_split is None:
            continue

        if subset_name == 'night':
            if curr_split == 'train':
                if curr_scene_name in splits.val_night:
                    curr_split = 'val'
        if subset_name == 'singapore':
            if curr_split == 'train':
                if curr_scene_name in splits.val_singapore:
                    curr_split = 'val'

        # filter for day/night
        if keyword:
            scene_description = nusc.get("scene", sample["scene_token"])["description"]
            if keyword.lower() in scene_description.lower():
                if keyword_action == 'exclude':
                    # skip sample
                    continue
            else:
                if keyword_action == 'filter':
                    # skip sample
                    continue

        if location:
            scene = nusc.get("scene", sample["scene_token"])
            if location not in nusc.get("log", scene['log_token'])['location']:
                continue

        lidar_token = sample["data"]["LIDAR_TOP"]
        cam_front_token = sample["data"]["CAM_FRONT"]
        calib_sen_token = nusc.get('sample_data', cam_front_token)['calibrated_sensor_token']
        
        lidar_path = osp.join(nusc.dataroot, nusc.get('sample_data', lidar_token)['filename'])
        label_path = osp.join(nusc.dataroot, nusc.get('lidarseg', lidar_token)['filename'])
        cam_path = osp.join(nusc.dataroot, nusc.get('sample_data', cam_front_token)['filename'])
        cam_intrinsic = np.array(nusc.get('calibrated_sensor', calib_sen_token)['camera_intrinsic'])

        print('{}/{} {} {}'.format(i + 1, len(nusc.sample), curr_scene_name, lidar_path))

        sd_rec_lidar = nusc.get('sample_data', sample['data']["LIDAR_TOP"])
        cs_record_lidar = nusc.get('calibrated_sensor',
                             sd_rec_lidar['calibrated_sensor_token'])
        pose_record_lidar = nusc.get('ego_pose', sd_rec_lidar['ego_pose_token'])
        sd_rec_cam = nusc.get('sample_data', sample['data']["CAM_FRONT"])
        cs_record_cam = nusc.get('calibrated_sensor',
                             sd_rec_cam['calibrated_sensor_token'])
        pose_record_cam = nusc.get('ego_pose', sd_rec_cam['ego_pose_token'])

        calib_infos = {
            "lidar2ego_translation": cs_record_lidar['translation'],
            "lidar2ego_rotation": cs_record_lidar['rotation'],
            "ego2global_translation_lidar": pose_record_lidar['translation'],
            "ego2global_rotation_lidar": pose_record_lidar['rotation'],
            "ego2global_translation_cam": pose_record_cam['translation'],
            "ego2global_rotation_cam": pose_record_cam['rotation'],
            "cam2ego_translation": cs_record_cam['translation'],
            "cam2ego_rotation": cs_record_cam['rotation'],
            "cam_intrinsic": cam_intrinsic,
        }

        # load lidar points
        pts = np.fromfile(lidar_path, dtype=np.float32, count=-1).reshape([-1, 5])[:, :3].T
        lbl = np.fromfile(label_path, dtype=np.uint8, count=-1)

        # map point cloud into front camera image
        pts_valid_flag, pts_cam_coord, pts_img = map_pointcloud_to_image(pts, (900, 1600, 3), calib_infos)
        # fliplr so that indexing is row, col and not col, row
        pts_img = np.ascontiguousarray(np.fliplr(pts_img))

        # only use lidar points in the front camera image
        pts = pts[:, pts_valid_flag]
        lbl = lbl[pts_valid_flag]

        # convert to relative path
        lidar_path = lidar_path.replace(root_dir + '/', '')
        cam_path = cam_path.replace(root_dir + '/', '')

        # transpose to yield shape (num_points, 3)
        pts = pts.T

        # append data to train, val or test list in pkl_dict
        data_dict = {
            'points': pts,
            'seg_labels': lbl,
            'points_img': pts_img,  # row, col format, shape: (num_points, 2)
            'lidar_path': lidar_path,
            'label_path': label_path,
            'camera_path': cam_path,
            "sample_token": sample["token"],
            "scene_name": curr_scene_name,
            "calib": calib_infos
        }
        pkl_dict[curr_split].append(data_dict)

    # save to pickle file
    save_dir = osp.join(out_dir, 'preprocess')
    os.makedirs(save_dir, exist_ok=True)
    for split_name in split_names:
        save_path = osp.join(save_dir, '{}{}.pkl'.format(split_name, '_' + subset_name if subset_name else ''))
        with open(save_path, 'wb') as f:
            pickle.dump(pkl_dict[split_name], f)
            print('Wrote preprocessed data to ' + save_path)


if __name__ == '__main__':
    root_dir = '/data/AmitRoyChowdhury/nuscenes'
    out_dir = '/data/AmitRoyChowdhury/xmuda/nuscenes_lidarseg'
    nusc = NuScenes(version='v1.0-trainval', dataroot=root_dir, verbose=True)
    # for faster debugging, the script can be run using the mini dataset
    # nusc = NuScenes(version='v1.0-mini', dataroot=root_dir, verbose=True)
    # We construct the splits by using the meta data of NuScenes:
    # USA/Singapore: We check if the location is Boston or Singapore.
    # Day/Night: We detect if "night" occurs in the scene description string.
    preprocess(nusc, ['train', 'test'], root_dir, out_dir, location='boston', subset_name='usa')
    preprocess(nusc, ['train', 'val', 'test'], root_dir, out_dir, location='singapore', subset_name='singapore')
    preprocess(nusc, ['train', 'test'], root_dir, out_dir, keyword='night', keyword_action='exclude', subset_name='day')
    preprocess(nusc, ['train', 'val', 'test'], root_dir, out_dir, keyword='night', keyword_action='filter', subset_name='night')
