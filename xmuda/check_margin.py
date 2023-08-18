#!/usr/bin/env python
import os
import os.path as osp
import argparse
import logging
import time
import socket
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions

from torch.nn.modules.module import _global_forward_hooks

from xmuda.common.utils.checkpoint import CheckpointerV2
from xmuda.common.utils.logger import setup_logger
from xmuda.common.utils.metric_logger import MetricLogger
from xmuda.common.utils.torch_util import set_random_seed
from xmuda.models.build import build_model_2d, build_model_3d
from xmuda.data.build import build_dataloader
from xmuda.data.utils.validate import validate


def parse_args():
    parser = argparse.ArgumentParser(description='xMUDA test')
    parser.add_argument(
        '--cfg',
        dest='config_file',
        default='',
        metavar='FILE',
        help='path to config file',
        type=str,
    )
    parser.add_argument('ckpt2d', type=str, help='path to checkpoint file of the 2D model')
    parser.add_argument('ckpt3d', type=str, help='path to checkpoint file of the 3D model')
    parser.add_argument('--source', action='store_true', help='Also calculate performance on source data')
    parser.add_argument(
        'opts',
        help='Modify config options using the command-line',
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    return args

def print_pretty(key, val, c=40):
    val_str = '{:6.3f}'.format(val)
    return '|' + ' '*(c-len(key)-len(val_str)-4) + key + ': ' + val_str + '|'

def entropy(preds):
    probs = F.softmax(preds, dim=1)
    log_probs = torch.log(probs)
    log_probs[torch.isnan(log_probs)] = 1
    log_probs[torch.isinf(log_probs)] = 1
    H = -torch.sum(probs * log_probs, dim=1)
    return H

class bn_2d_hook():
    def __init__(self):
        self._mean = None
        self._var = None 
        self._N = 0

    def __call__(self, m, i, o):
        m = torch.mean(o.detach().clone(), dim=(0, 2, 3))
        v = torch.var(o.detach().clone(), dim=(0, 2, 3))
        c = o.numel() / o.shape[1] 
        if self._N:
            self._mean = (self._N/(self._N + c)) * self._mean + (c/(self._N + c)) * m
            self._var = (self._N/(self._N + c)) * self._var + (c/(self._N + c)) * v + ((c * self._N)/((c+self._mean)**2)) * (self._mean - m)**2
            self._N += c
        else:
            self._mean = m
            self._var = v
            self._N = c 

    def reset(self):
        self._mean = None
        self._var = None 
        self._N = 0

    @property
    def mean(self):
        return self._mean

    @property
    def var(self):
        return self._var

class bn_3d_hook():
    def __init__(self):
        self._mean = None
        self._var = None 
        self._N = 0

    def __call__(self, m, i, o):
        m = torch.mean(o.features.detach().clone(), dim=(0))
        v = torch.var(o.features.detach().clone(), dim=(0))
        c = o.features.shape[0] 
        if self._N:
            self._mean = (self._N/(self._N + c)) * self._mean + (c/(self._N + c)) * m
            self._var = (self._N/(self._N + c)) * self._var + (c/(self._N + c)) * v + ((c * self._N)/((c+self._mean)**2)) * (self._mean - m)**2
            self._N += c
        else:
            self._mean = m
            self._var = v
            self._N = c 
    
    def reset(self):
        self._mean = None
        self._var = None 
        self._N = 0

    @property
    def mean(self):
        return self._mean

    @property
    def var(self):
        return self._var

def KL_div(m1, m2, v1, v2):
    n1 = distributions.normal.Normal(m1, v1)
    n2 = distributions.normal.Normal(m2, v2)

    return distributions.kl.kl_divergence(n1, n2).sum().item()

def norm2(m1, m2):
    return torch.sqrt(((m1-m2)**2).sum()).item()

def check_stats(cfg, model_2d, model_3d, dataloader, logger):
    min_H_2d = 1e10
    min_H_3d = 1e10
    max_H_2d = -1
    max_H_3d = -1
    avg_H_2d = 0
    avg_H_3d = 0

    num_classes = len(dataloader.dataset.class_names)
    min_margin_2d = 1e10 * torch.ones(1, num_classes, dtype=torch.float32, device=torch.device('cuda'))
    min_margin_3d = 1e10 * torch.ones(1, num_classes, dtype=torch.float32, device=torch.device('cuda'))
    max_margin_2d = -1e10 * torch.ones(1, num_classes, dtype=torch.float32, device=torch.device('cuda'))
    max_margin_3d = -1e10 * torch.ones(1, num_classes, dtype=torch.float32, device=torch.device('cuda'))
    avg_margin_2d = torch.zeros(num_classes, dtype=torch.float32, device=torch.device('cuda')) 
    avg_margin_3d = torch.zeros(num_classes, dtype=torch.float32, device=torch.device('cuda'))

    num_items = 0

    with torch.no_grad():
        for iteration, data_batch in enumerate(dataloader):
            if 'SCN' in cfg.DATASET_TARGET.TYPE:
                data_batch['x'][1] = data_batch['x'][1].cuda()
                data_batch['img'] = data_batch['img'].cuda()
            else:
                raise NotImplementedErrorj

            preds_2d = model_2d(data_batch)
            preds_3d = model_3d(data_batch) if model_3d else None

            H_2d = entropy(preds_2d['seg_logit'])
            H_3d = entropy(preds_3d['seg_logit'])

            min_H_2d = min(min_H_2d, torch.min(H_2d).item())
            min_H_3d = min(min_H_3d, torch.min(H_3d).item())
            max_H_2d = max(max_H_2d, torch.max(H_2d).item())
            max_H_3d = max(max_H_3d, torch.max(H_3d).item())
            avg_H_2d += torch.sum(H_2d).item()
            avg_H_3d += torch.sum(H_3d).item()

            margin_2d = torch.abs(preds_2d['seg_logit'])
            margin_3d = torch.abs(preds_3d['seg_logit'])
            min_margin_2d = torch.min(torch.cat((min_margin_2d, margin_2d), 0), 0, True)[0]
            min_margin_3d = torch.min(torch.cat((min_margin_3d, margin_3d), 0), 0, True)[0]
            max_margin_2d = torch.max(torch.cat((max_margin_2d, margin_2d), 0), 0, True)[0]
            max_margin_3d = torch.max(torch.cat((max_margin_3d, margin_3d), 0), 0, True)[0]
            avg_margin_2d += torch.sum(margin_2d, 0) 
            avg_margin_3d += torch.sum(margin_3d, 0) 
            num_items += preds_2d['seg_logit'].shape[0]

    KL_divs_2d = [KL_div(model_2d.net_2d.dec_t_conv_stage2[0]._forward_hooks[2].mean,
                         model_2d.net_2d.dec_t_conv_stage2[1].running_mean,
                         model_2d.net_2d.dec_t_conv_stage2[0]._forward_hooks[2].var,
                         model_2d.net_2d.dec_t_conv_stage2[1].running_var),
                  KL_div(model_2d.net_2d.dec_t_conv_stage3[0]._forward_hooks[1].mean,
                         model_2d.net_2d.dec_t_conv_stage3[1].running_mean,
                         model_2d.net_2d.dec_t_conv_stage3[0]._forward_hooks[1].var,
                         model_2d.net_2d.dec_t_conv_stage3[1].running_var),
                  KL_div(model_2d.net_2d.dec_t_conv_stage4[0]._forward_hooks[0].mean,
                         model_2d.net_2d.dec_t_conv_stage4[1].running_mean,
                         model_2d.net_2d.dec_t_conv_stage4[0]._forward_hooks[0].var,
                         model_2d.net_2d.dec_t_conv_stage4[1].running_var)]
    norm2_2d = [norm2(model_2d.net_2d.dec_t_conv_stage2[0]._forward_hooks[2].mean,
                      model_2d.net_2d.dec_t_conv_stage2[1].running_mean),
                norm2(model_2d.net_2d.dec_t_conv_stage3[0]._forward_hooks[1].mean,
                      model_2d.net_2d.dec_t_conv_stage3[1].running_mean),
                norm2(model_2d.net_2d.dec_t_conv_stage4[0]._forward_hooks[0].mean,
                      model_2d.net_2d.dec_t_conv_stage4[1].running_mean)]
    KL_divs_3d = [KL_div(model_3d.net_3d.sparseModel[2][3]._forward_hooks[3].mean,
                         model_3d.net_3d.sparseModel[3].running_mean,
                         model_3d.net_3d.sparseModel[2][3]._forward_hooks[3].var,
                         model_3d.net_3d.sparseModel[3].running_var),
                  KL_div(model_3d.net_3d.sparseModel[2][1][1][2][3]._forward_hooks[4].mean,
                         model_3d.net_3d.sparseModel[2][1][1][3].running_mean,
                         model_3d.net_3d.sparseModel[2][1][1][2][3]._forward_hooks[4].var,
                         model_3d.net_3d.sparseModel[2][1][1][3].running_var),
                  KL_div(model_3d.net_3d.sparseModel[2][1][1][2][1][1][2][3]._forward_hooks[5].mean,
                         model_3d.net_3d.sparseModel[2][1][1][2][1][1][3].running_mean,
                         model_3d.net_3d.sparseModel[2][1][1][2][1][1][2][3]._forward_hooks[5].var,
                         model_3d.net_3d.sparseModel[2][1][1][2][1][1][3].running_var)]
    norm2_3d = [norm2(model_3d.net_3d.sparseModel[2][3]._forward_hooks[3].mean,
                      model_3d.net_3d.sparseModel[3].running_mean),
                norm2(model_3d.net_3d.sparseModel[2][1][1][2][3]._forward_hooks[4].mean,
                      model_3d.net_3d.sparseModel[2][1][1][3].running_mean),
                norm2(model_3d.net_3d.sparseModel[2][1][1][2][1][1][2][3]._forward_hooks[5].mean,
                      model_3d.net_3d.sparseModel[2][1][1][2][1][1][3].running_mean)]
    
    logger.info('----------------------------------------')
    logger.info('|            2D Statistics              |')
    logger.info('----------------------------------------')
    logger.info(print_pretty('Minimum Entropy', min_H_2d))
    logger.info(print_pretty('Maximum Entropy', max_H_2d))
    logger.info(print_pretty('Average Entropy', avg_H_2d/num_items))
    logger.info(print_pretty('Average Margin', torch.mean(avg_margin_2d/num_items).item()))
    logger.info(print_pretty('Average Min Margin', torch.mean(min_margin_2d).item()))
    logger.info(print_pretty('Minimum Min Margin', torch.min(min_margin_2d).item()))
    logger.info(print_pretty('Average Max Margin', torch.mean(max_margin_2d).item()))
    logger.info(print_pretty('Maximum Max Margin', torch.max(max_margin_2d).item()))
    logger.info(print_pretty('KL_div 0', KL_divs_2d[0]))
    logger.info(print_pretty('KL_div 1', KL_divs_2d[1]))
    logger.info(print_pretty('KL_div 2', KL_divs_2d[2]))
    logger.info(print_pretty('Euclidean 0', norm2_2d[0]))
    logger.info(print_pretty('Euclidean 1', norm2_2d[1]))
    logger.info(print_pretty('Euclidean 2', norm2_2d[2]))
    logger.info('----------------------------------------')
    logger.info('----------------------------------------')
    logger.info('|            3D Statistics              |')
    logger.info('----------------------------------------')
    logger.info(print_pretty('Minimum Entropy', min_H_3d))
    logger.info(print_pretty('Maximum Entropy', max_H_3d))
    logger.info(print_pretty('Average Entropy', avg_H_3d/num_items))
    logger.info(print_pretty('Average Margin', torch.mean(avg_margin_3d/num_items).item()))
    logger.info(print_pretty('Average Min Margin', torch.mean(min_margin_3d).item()))
    logger.info(print_pretty('Minimum Min Margin', torch.min(min_margin_3d).item()))
    logger.info(print_pretty('Average Max Margin', torch.mean(max_margin_3d).item()))
    logger.info(print_pretty('Maximum Max Margin', torch.max(max_margin_3d).item()))
    logger.info(print_pretty('KL_div 0', KL_divs_3d[0]))
    logger.info(print_pretty('KL_div 1', KL_divs_3d[1]))
    logger.info(print_pretty('KL_div 2', KL_divs_3d[2]))
    logger.info(print_pretty('Euclidean 0', norm2_3d[0]))
    logger.info(print_pretty('Euclidean 1', norm2_3d[1]))
    logger.info(print_pretty('Euclidean 2', norm2_3d[2]))
    logger.info('----------------------------------------')

def check(cfg, args, output_dir=''):
    logger = logging.getLogger('xmuda.test')

    # build 2d model
    model_2d = build_model_2d(cfg)[0]

    # build 3d model
    model_3d = build_model_3d(cfg)[0]

    model_2d = model_2d.cuda()
    model_3d = model_3d.cuda()

    # build checkpointer
    checkpointer_2d = CheckpointerV2(model_2d, save_dir=output_dir, logger=logger)
    weight_path = args.ckpt2d.replace('@', output_dir)
    checkpointer_2d.load(weight_path, resume=False)
    
    checkpointer_3d = CheckpointerV2(model_3d, save_dir=output_dir, logger=logger)
    weight_path = args.ckpt3d.replace('@', output_dir)
    checkpointer_3d.load(weight_path, resume=False)

    if args.source:
        source_dataloader = build_dataloader(cfg, mode='test', domain='source')
    target_dataloader = build_dataloader(cfg, mode='val', domain='target')

    set_random_seed(cfg.RNG_SEED)
    metric_logger = MetricLogger(delimiter='  ')
    model_2d.eval()
    model_3d.eval()

    handles = []
    handles.append(model_2d.net_2d.dec_t_conv_stage4[0].register_forward_hook(bn_2d_hook()))
    handles.append(model_2d.net_2d.dec_t_conv_stage3[0].register_forward_hook(bn_2d_hook()))
    handles.append(model_2d.net_2d.dec_t_conv_stage2[0].register_forward_hook(bn_2d_hook()))
    
    handles.append(model_3d.net_3d.sparseModel[2][3].register_forward_hook(bn_3d_hook()))
    handles.append(model_3d.net_3d.sparseModel[2][1][1][2][3].register_forward_hook(bn_3d_hook()))
    handles.append(model_3d.net_3d.sparseModel[2][1][1][2][1][1][2][3].register_forward_hook(bn_3d_hook()))
    
    for handle in handles:
        hook = handle.hooks_dict_ref()[handle.id]
        print(hook._mean, hook._var, hook._N) 

    if args.source:
        logger.info('Source Test Set')
        check_stats(cfg, model_2d, model_3d, source_dataloader, logger)    
        for handle in handles:
            hook = handle.hooks_dict_ref()[handle.id]
            hook.reset()
   #     checkpointer_2d = CheckpointerV2(model_2d, save_dir=output_dir, logger=logger)
   #     weight_path = args.ckpt2d.replace('@', output_dir)
   #     checkpointer_2d.load(weight_path, resume=False)
        
   #     checkpointer_3d = CheckpointerV2(model_3d, save_dir=output_dir, logger=logger)
   #     weight_path = args.ckpt3d.replace('@', output_dir)
   #     checkpointer_3d.load(weight_path, resume=False)
    
    for handle in handles:
        hook = handle.hooks_dict_ref()[handle.id]
        print(hook._mean, hook._var, hook._N) 

    logger.info('Target Validation Set')
    check_stats(cfg, model_2d, model_3d, target_dataloader, logger)

def main():
    args = parse_args()

    # load the configuration
    # import on-the-fly to avoid overwriting cfg
    from xmuda.common.config import purge_cfg
    from xmuda.config.xmuda import cfg
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    purge_cfg(cfg)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    # replace '@' with config path
    if output_dir:
        config_path = osp.splitext(args.config_file)[0]
        output_dir = output_dir.replace('@', config_path.replace('configs/', ''))
        if not osp.isdir(output_dir):
            warnings.warn('Make a new directory: {}'.format(output_dir))
            os.makedirs(output_dir)

    # run name
    timestamp = time.strftime('%m-%d_%H-%M-%S')
    hostname = socket.gethostname()
    run_name = '{:s}.{:s}'.format(timestamp, hostname)

    logger = setup_logger('xmuda', output_dir, comment='test.{:s}'.format(run_name))
    logger.info('{:d} GPUs available'.format(torch.cuda.device_count()))
    logger.info(args)

    logger.info('Loaded configuration file {:s}'.format(args.config_file))
    logger.info('Running with config:\n{}'.format(cfg))

    assert cfg.MODEL_2D.DUAL_HEAD == cfg.MODEL_3D.DUAL_HEAD
    check(cfg, args, output_dir)


if __name__ == '__main__':
    main()
