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
        return getattr(torch.optim, name)(
            model.parameters(),
            lr=cfg.OPTIMIZER.BASE_LR,
            weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY,
            **cfg.OPTIMIZER.get(name, dict()),
        )
    else:
        raise ValueError('Unsupported type of optimizer.')

def build_optimizer_nrc(cfg, model):
    name = cfg.OPTIMIZER.TYPE
    if name == '':
        warnings.warn('No optimizer is built.')
        return None
    elif hasattr(torch.optim, name):
        return getattr(torch.optim, name)(
            model.nrc_parameters(),
            lr=cfg.OPTIMIZER.BASE_LR,
            weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY,
            **cfg.OPTIMIZER.get(name, dict()),
        )
    else:
        raise ValueError('Unsupported type of optimizer.')

def build_generator_optimizer(cfg, model):
    name = cfg.GEN_OPTIMIZER.TYPE
    if name == '':
        warnings.warn('No optimizer is built.')
        return None
    elif hasattr(torch.optim, name):
        return getattr(torch.optim, name)(
            model.parameters(),
            lr=cfg.GEN_OPTIMIZER.BASE_LR,
            weight_decay=cfg.GEN_OPTIMIZER.WEIGHT_DECAY,
            **cfg.GEN_OPTIMIZER.get(name, dict()),
        )
    else:
        raise ValueError('Unsupported type of optimizer.')

def build_discriminator_optimizer(cfg, model):
    name = cfg.DIS_OPTIMIZER.TYPE
    if name == '':
        warnings.warn('No optimizer is built.')
        return None
    elif hasattr(torch.optim, name):
        return getattr(torch.optim, name)(
            model.parameters(),
            lr=cfg.DIS_OPTIMIZER.BASE_LR,
            weight_decay=cfg.DIS_OPTIMIZER.WEIGHT_DECAY,
            **cfg.DIS_OPTIMIZER.get(name, dict()),
        )
    else:
        raise ValueError('Unsupported type of optimizer.')

def build_autoencoder_optimizer(cfg, model):
    name = cfg.AE_OPTIMIZER.TYPE
    if name == '':
        warnings.warn('No optimizer is built.')
        return None
    elif hasattr(torch.optim, name):
        return getattr(torch.optim, name)(
            model.parameters(),
            lr=cfg.AE_OPTIMIZER.BASE_LR,
            weight_decay=cfg.AE_OPTIMIZER.WEIGHT_DECAY,
            **cfg.AE_OPTIMIZER.get(name, dict()),
        )
    else:
        return ValueError('Unsupported type of optimizer.')

def build_encoder_optimizer(cfg, model):
    name = cfg.ENCODER_OPTIMIZER.TYPE
    if name == '':
        warnings.warn('No optimizer is built.')
        return None
    elif hasattr(torch.optim, name):
        return getattr(torch.optim, name)(
            model.parameters(),
            lr=cfg.ENCODER_OPTIMIZER.BASE_LR,
            weight_decay=cfg.ENCODER_OPTIMIZER.WEIGHT_DECAY,
            **cfg.ENCODER_OPTIMIZER.get(name, dict()),
        )
    else:
        return ValueError('Unsupported type of optimizer.')

def build_decoder_optimizer(cfg, model):
    name = cfg.DECODER_OPTIMIZER.TYPE
    if name == '':
        warnings.warn('No optimizer is built.')
        return None
    elif hasattr(torch.optim, name):
        return getattr(torch.optim, name)(
            model.parameters(),
            lr=cfg.DECODER_OPTIMIZER.BASE_LR,
            weight_decay=cfg.DECODER_OPTIMIZER.WEIGHT_DECAY,
            **cfg.DECODER_OPTIMIZER.get(name, dict()),
        )
    else:
        return ValueError('Unsupported type of optimizer.')

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

def build_autoencoder_scheduler(cfg, optimizer):
    name = cfg.AE_SCHEDULER.TYPE
    if name == '':
        warnings.warn('No scheduler is built.')
        return None
    elif hasattr(torch.optim.lr_scheduler, name):
        scheduler = getattr(torch.optim.lr_scheduler, name)(
            optimizer,
            **cfg.SCHEDULER.get(name, dict()),
        )
    else:
        raise ValueError('Unsupported type of scheduler.')

    # clip learning rate
    if cfg.AE_SCHEDULER.CLIP_LR > 0.0:
        print('Learning rate is clipped to {}'.format(cfg.SCHEDULER.CLIP_LR))
        scheduler = cliplr(scheduler, min_lr=cfg.SCHEDULER.CLIP_LR)
    return scheduler

def build_encoder_scheduler(cfg, optimizer):
    name = cfg.ENCODER_SCHEDULER.TYPE
    if name == '':
        warnings.warn('No scheduler is built.')
        return None
    elif hasattr(torch.optim.lr_scheduler, name):
        scheduler = getattr(torch.optim.lr_scheduler, name)(
            optimizer,
            **cfg.SCHEDULER.get(name, dict()),
        )
    else:
        raise ValueError('Unsupported type of scheduler.')

    # clip learning rate
    if cfg.ENCODER_SCHEDULER.CLIP_LR > 0.0:
        print('Learning rate is clipped to {}'.format(cfg.SCHEDULER.CLIP_LR))
        scheduler = cliplr(scheduler, min_lr=cfg.SCHEDULER.CLIP_LR)
    return scheduler

def build_decoder_scheduler(cfg, optimizer):
    name = cfg.DECODER_SCHEDULER.TYPE
    if name == '':
        warnings.warn('No scheduler is built.')
        return None
    elif hasattr(torch.optim.lr_scheduler, name):
        scheduler = getattr(torch.optim.lr_scheduler, name)(
            optimizer,
            **cfg.SCHEDULER.get(name, dict()),
        )
    else:
        raise ValueError('Unsupported type of scheduler.')

    # clip learning rate
    if cfg.DECODER_SCHEDULER.CLIP_LR > 0.0:
        print('Learning rate is clipped to {}'.format(cfg.SCHEDULER.CLIP_LR))
        scheduler = cliplr(scheduler, min_lr=cfg.SCHEDULER.CLIP_LR)
    return scheduler

def build_generator_scheduler(cfg, optimizer):
    name = cfg.GEN_SCHEDULER.TYPE
    if name == '':
        warnings.warn('No scheduler is built.')
        return None
    elif hasattr(torch.optim.lr_scheduler, name):
        scheduler = getattr(torch.optim.lr_scheduler, name)(
            optimizer,
            **cfg.GEN_SCHEDULER.get(name, dict()),
        )
    else:
        raise ValueError('Unsupported type of scheduler.')

    # clip learning rate
    if cfg.GEN_SCHEDULER.CLIP_LR > 0.0:
        print('Learning rate is clipped to {}'.format(cfg.GEN_SCHEDULER.CLIP_LR))
        scheduler = cliplr(scheduler, min_lr=cfg.GEN_SCHEDULER.CLIP_LR)

    return scheduler

def build_discriminator_scheduler(cfg, optimizer):
    name = cfg.DIS_SCHEDULER.TYPE
    if name == '':
        warnings.warn('No scheduler is built.')
        return None
    elif hasattr(torch.optim.lr_scheduler, name):
        scheduler = getattr(torch.optim.lr_scheduler, name)(
            optimizer,
            **cfg.DIS_SCHEDULER.get(name, dict()),
        )
    else:
        raise ValueError('Unsupported type of scheduler.')

    # clip learning rate
    if cfg.DIS_SCHEDULER.CLIP_LR > 0.0:
        print('Learning rate is clipped to {}'.format(cfg.DIS_SCHEDULER.CLIP_LR))
        scheduler = cliplr(scheduler, min_lr=cfg.DIS_SCHEDULER.CLIP_LR)

    return scheduler
