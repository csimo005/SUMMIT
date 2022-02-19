import numpy as np
import logging
import time
import os

import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt

from xmuda.data.utils.evaluate import Evaluator
from xmuda.data.utils.visualize import *

def validate(cfg,
             model_2d,
             model_3d,
             dataloader,
             val_metric_logger,
             pselab_path=None,
             img_path=None,
             video_path=None):
    logger = logging.getLogger('xmuda.validate')
    logger.info('Validation')

    if img_path or video_path:
        model_2d.full_img = True

    # evaluator
    class_names = dataloader.dataset.class_names
    evaluator_2d = Evaluator(class_names)
    evaluator_3d = Evaluator(class_names) if model_3d else None
    evaluator_ensemble = Evaluator(class_names) if model_3d else None

    pselab_data_list = []

    end = time.time()
    iter = 0
    with torch.no_grad():
        for iteration, data_batch in enumerate(dataloader):
            data_time = time.time() - end
            # copy data from cpu to gpu
            if 'SCN' in cfg.DATASET_TARGET.TYPE:
                data_batch['x'][1] = data_batch['x'][1].cuda()
                data_batch['seg_label'] = data_batch['seg_label'].cuda()
                data_batch['img'] = data_batch['img'].cuda()
            else:
                raise NotImplementedError

            # predict
            preds_2d = model_2d(data_batch)
            preds_3d = model_3d(data_batch) if model_3d else None

            pred_label_voxel_2d = preds_2d['seg_logit'].argmax(1).cpu().numpy()
            if model_2d.dual_head:
                pred_label_voxel_2d_3d = preds_2d['seg_logit2'].argmax(1).cpu().numpy()
                
            pred_label_voxel_3d = preds_3d['seg_logit'].argmax(1).cpu().numpy() if model_3d else None
            if model_3d.dual_head: 
                pred_label_voxel_3d_2d = preds_3d['seg_logit2'].argmax(1).cpu().numpy() if model_3d else None

            # softmax average (ensembling)
            probs_2d = F.softmax(preds_2d['seg_logit'], dim=1)
            probs_3d = F.softmax(preds_3d['seg_logit'], dim=1) if model_3d else None
            feats_2d = preds_2d['feats']
            feats_3d = preds_3d['feats'] if model_3d else None
            pred_label_voxel_ensemble = (probs_2d + probs_3d).argmax(1).cpu().numpy() if model_3d else None

            # get original point cloud from before voxelization
            seg_label = data_batch['orig_seg_label']
            points_idx = data_batch['orig_points_idx']
            # loop over batch
            left_idx = 0
            for batch_ind in range(len(seg_label)):
                curr_points_idx = points_idx[batch_ind]
                # check if all points have predictions (= all voxels inside receptive field)
                assert np.all(curr_points_idx)

                curr_seg_label = seg_label[batch_ind]
                right_idx = left_idx + curr_points_idx.sum()
                pred_label_2d = pred_label_voxel_2d[left_idx:right_idx]
                pred_label_3d = pred_label_voxel_3d[left_idx:right_idx] if model_3d else None
                pred_label_ensemble = pred_label_voxel_ensemble[left_idx:right_idx] if model_3d else None

                # evaluate
                evaluator_2d.update(pred_label_2d, curr_seg_label)
                if model_3d:
                    evaluator_3d.update(pred_label_3d, curr_seg_label)
                    evaluator_ensemble.update(pred_label_ensemble, curr_seg_label)

                if pselab_path is not None:
                    assert np.all(pred_label_2d >= 0)
                    curr_probs_2d = probs_2d[left_idx:right_idx]
                    curr_probs_3d = probs_3d[left_idx:right_idx] if model_3d else None
                    curr_feats_2d = feats_2d[left_idx:right_idx]
                    curr_feats_3d = feats_3d[left_idx:right_idx] if model_3d else None
                    pselab_data_list.append({
                        'probs_2d': curr_probs_2d.cpu().numpy(),
                        'pseudo_label_2d': pred_label_2d.astype(np.uint8),
                        'features_2d': curr_feats_2d.cpu().numpy(),
                        'probs_3d': curr_probs_3d.cpu().numpy() if model_3d else None,
                        'pseudo_label_3d': pred_label_3d.astype(np.uint8) if model_3d else None,
                        'features_3d': curr_feats_3d.cpu().numpy() if model_3d else None
                    })

                #draw images, first iteration only.
                if (img_path and iteration==0) or video_path:
                    print(data_batch['img_path'][batch_ind])
                    
                    img = plt.imread(data_batch['img_path'][batch_ind]) 
                    img_indices = data_batch['img_indices'][batch_ind]
                    seg_labels = pred_label_voxel_2d[left_idx:right_idx]
                    save_pth = os.path.join(img_path, '2d_ptcloud_{}.png'.format(iter))
                    draw_points_image_labels(img, img_indices, seg_labels, save_pth=save_pth, color_palette_type=cfg.DATASET_TARGET.TYPE[:-3])
                    draw_points_pred_gt(img, img_indices, pred_label_voxel_2d[left_idx:right_idx], data_batch['seg_label'][left_idx:right_idx].cpu().numpy(), save_pth='2d_correct.png')
               
                    if model_2d.dual_head:
                        seg_labels = pred_label_voxel_2d_3d[left_idx:right_idx]
                        save_pth = os.path.join(img_path, '2d_3d_ptcloud_{}.png'.format(iter))
                        draw_points_image_labels(img, img_indices, seg_labels, save_pth=save_pth, color_palette_type=cfg.DATASET_TARGET.TYPE[:-3])
                    
                    seg_labels = pred_label_voxel_3d[left_idx:right_idx]
                    save_pth = os.path.join(img_path, '3d_ptcloud_{}.png'.format(iter))
                    draw_points_image_labels(img, img_indices, seg_labels, save_pth=save_pth, color_palette_type=cfg.DATASET_TARGET.TYPE[:-3])
                    draw_points_pred_gt(img, img_indices, pred_label_voxel_3d[left_idx:right_idx], data_batch['seg_label'][left_idx:right_idx].cpu().numpy(), save_pth='3d_correct.png')
                   
                    if model_3d.dual_head: 
                        seg_labels = pred_label_voxel_3d_2d[left_idx:right_idx]
                        save_pth = os.path.join(img_path, '3d_2d_ptcloud_{}.png'.format(iter))
                        draw_points_image_labels(img, img_indices, seg_labels, save_pth=save_pth, color_palette_type=cfg.DATASET_TARGET.TYPE[:-3])
                    
                    seg_labels = pred_label_voxel_ensemble[left_idx:right_idx]
                    save_pth = os.path.join(img_path, 'softmax_ptcloud_{}.png'.format(iter))
                    draw_points_image_labels(img, img_indices, seg_labels, save_pth=save_pth, color_palette_type=cfg.DATASET_TARGET.TYPE[:-3])
                    
                    seg_labels = data_batch['seg_label'][left_idx:right_idx].cpu().numpy()
                    save_pth = os.path.join(img_path, 'gt_ptcloud_{}.png'.format(iter))
                    draw_points_image_labels(img, img_indices, seg_labels, save_pth=save_pth, color_palette_type=cfg.DATASET_TARGET.TYPE[:-3])

                    draw_points_pred_agree(img, img_indices, pred_label_voxel_2d[left_idx:right_idx], pred_label_voxel_3d[left_idx:right_idx], save_pth='agreement.png')
                    
                    img = preds_2d['full_logit'][batch_ind].argmax(0).cpu().numpy()
                    save_pth = os.path.join(img_path, '2d_fullimage_{}.png'.format(iter))
                    print(save_pth)
                    draw_seg_image(img, save_pth=save_pth, color_palette_type=cfg.DATASET_TARGET.TYPE[:-3])

                    iter += 1
                    exit()
                    

                left_idx = right_idx
                    

            seg_loss_2d = F.cross_entropy(preds_2d['seg_logit'], data_batch['seg_label'])
            seg_loss_3d = F.cross_entropy(preds_3d['seg_logit'], data_batch['seg_label']) if model_3d else None
            val_metric_logger.update(seg_loss_2d=seg_loss_2d)
            if seg_loss_3d is not None:
                val_metric_logger.update(seg_loss_3d=seg_loss_3d)

            batch_time = time.time() - end
            val_metric_logger.update(time=batch_time, data=data_time)
            end = time.time()

            if not video_path:
                model_2d.full_img = False

            # log
            cur_iter = iteration + 1
            if cur_iter == 1 or (cfg.VAL.LOG_PERIOD > 0 and cur_iter % cfg.VAL.LOG_PERIOD == 0):
                logger.info(
                    val_metric_logger.delimiter.join(
                        [
                            'iter: {iter}/{total_iter}',
                            '{meters}',
                            'max mem: {memory:.0f}',
                        ]
                    ).format(
                        iter=cur_iter,
                        total_iter=len(dataloader),
                        meters=str(val_metric_logger),
                        memory=torch.cuda.max_memory_allocated() / (1024.0 ** 2),
                    )
                )

        val_metric_logger.update(seg_iou_2d=evaluator_2d.overall_iou)
        if evaluator_3d is not None:
            val_metric_logger.update(seg_iou_3d=evaluator_3d.overall_iou)
        eval_list = [('2D', evaluator_2d)]
        if model_3d:
            eval_list.extend([('3D', evaluator_3d), ('2D+3D', evaluator_ensemble)])
        for modality, evaluator in eval_list:
            logger.info('{} overall accuracy={:.2f}%'.format(modality, 100.0 * evaluator.overall_acc))
            logger.info('{} overall IOU={:.2f}'.format(modality, 100.0 * evaluator.overall_iou))
            logger.info('{} class-wise segmentation accuracy and IoU.\n{}'.format(modality, evaluator.print_table()))

        model_2d.full_img = False
        if pselab_path is not None:
            np.save(pselab_path, pselab_data_list)
            logger.info('Saved pseudo label data to {}'.format(pselab_path))
