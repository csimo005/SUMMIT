"""Build optimizers and schedulers"""
import warnings
import torch
from .lr_scheduler import ClipLR


def build_optimizer(cfg, model):
    name = cfg.OPTIMIZER.TYPE
    if name == '':
        warnings.warn('No optimizer is built.')
        return None
    elif hasattr(torch.optim, name):
        parameters = [{'params': model.feat_encoder_parameters(),
                       'lr': cfg.OPTIMIZER.FEAT_ENCODER_LR if cfg.OPTIMIZER.FEAT_ENCODER_LR > -1 else cfg.OPTIMIZER.BASE_LR,
                       'weight_decay': cfg.OPTIMIZER.FEAT_ENCODER_WEIGHT_DECAY if cfg.OPTIMIZER.FEAT_ENCODER_WEIGHT_DECAY > -1 else cfg.OPTIMIZER.WEIGHT_DECAY},
                      {'params': model.classifier_parameters(),
                       'lr': cfg.OPTIMIZER.CLASSIFIER_LR if cfg.OPTIMIZER.CLASSIFIER_LR > -1 else cfg.OPTIMIZER.BASE_LR,
                       'weight_decay': cfg.OPTIMIZER.CLASSIFIER_WEIGHT_DECAY if cfg.OPTIMIZER.CLASSIFIER_WEIGHT_DECAY > -1 else cfg.OPTIMIZER.WEIGHT_DECAY}]
        xmuda_params = model.xmuda_classifier_parameters()
        if not xmuda_params is None:
            parameters.append({'params': model.xmuda_classifier_parameters(),
                               'lr': cfg.OPTIMIZER.XMUDA_CLASSIFIER_LR if cfg.OPTIMIZER.XMUDA_CLASSIFIER_LR > -1 else cfg.OPTIMIZER.BASE_LR,
                               'weight_decay': cfg.OPTIMIZER.XMUDA_CLASSIFIER_WEIGHT_DECAY if cfg.OPTIMIZER.XMUDA_CLASSIFIER_WEIGHT_DECAY > -1 else cfg.OPTIMIZER.WEIGHT_DECAY})
        return getattr(torch.optim, name)(
            parameters, 
            lr=cfg.OPTIMIZER.BASE_LR,
            weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY,
            **cfg.OPTIMIZER.get(name, dict()),
        )
    else:
        raise ValueError('Unsupported type of optimizer.')

def build_scheduler(cfg, optimizer):
    name = cfg.SCHEDULER.TYPE
    if name == '':
        warnings.warn('no scheduler is built.')
        return None
    elif hasattr(torch.optim.lr_scheduler, name):
        scheduler = getattr(torch.optim.lr_scheduler, name)(
            optimizer,
            **cfg.SCHEDULER.get(name, dict()),
        )
    else:
        raise valueerror('unsupported type of scheduler.')

    # clip learning rate
    if cfg.SCHEDULER.CLIP_LR > 0.0:
        print('learning rate is clipped to {}'.format(cfg.SCHEDULER.CLIP_LR))
        scheduler = cliplr(scheduler, min_lr=cfg.SCHEDULER.CLIP_LR) 
    return scheduler
