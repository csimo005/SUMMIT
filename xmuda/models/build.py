from xmuda.models.xmuda_arch import Net2DSeg, Net3DSeg
from xmuda.models.cyclegan_arch import Generator, Discriminator
from xmuda.models.autoencoder_arch import AutoEncoder
from xmuda.models.metric import SegIoU, ConfMatrix


def build_model_2d(cfg):
    model = Net2DSeg(num_classes=cfg.MODEL_2D.NUM_CLASSES,
                     backbone_2d=cfg.MODEL_2D.TYPE,
                     backbone_2d_kwargs=cfg.MODEL_2D[cfg.MODEL_2D.TYPE],
                     dual_head=cfg.MODEL_2D.DUAL_HEAD
                     )
    train_metric = SegIoU(cfg.MODEL_2D.NUM_CLASSES, name='seg_iou_2d')
    return model, train_metric


def build_model_3d(cfg):
    model = Net3DSeg(num_classes=cfg.MODEL_3D.NUM_CLASSES,
                     backbone_3d=cfg.MODEL_3D.TYPE,
                     backbone_3d_kwargs=cfg.MODEL_3D[cfg.MODEL_3D.TYPE],
                     dual_head=cfg.MODEL_3D.DUAL_HEAD
                     )
    train_metric = SegIoU(cfg.MODEL_3D.NUM_CLASSES, name='seg_iou_3d')
    return model, train_metric

def build_generators(cfg):
    gen_2d_3d = Generator(cfg.MODEL_2D.FEAT_DIM, cfg.MODEL_3D.FEAT_DIM, cfg.GENERATOR.EMBED_DIM)
    gen_3d_2d = Generator(cfg.MODEL_3D.FEAT_DIM, cfg.MODEL_2D.FEAT_DIM, cfg.GENERATOR.EMBED_DIM)
    return gen_2d_3d, gen_3d_2d, None, None 

def build_discriminators(cfg):
    dis_2d = Discriminator(cfg.MODEL_2D.FEAT_DIM)
    dis_3d = Discriminator(cfg.MODEL_3D.FEAT_DIM)

    dis_2d_metric = ConfMatrix(name='Discriminator 2D', inverse_labels=cfg.TRAIN.XMUDA.LABELING.inverse_label)
    dis_3d_metric = ConfMatrix(name='Discriminator 3D', inverse_labels=cfg.TRAIN.XMUDA.LABELING.inverse_label)
    return dis_2d, dis_3d, dis_2d_metric, dis_3d_metric

def build_autoencoder(cfg):
    ae_2d_3d = AutoEncoder(cfg.MODEL_2D.FEAT_DIM, cfg.MODEL_3D.FEAT_DIM, cfg.AE.EMBED_DIM, cfg.AE.HIDDEN_LAYERS)
    ae_3d_2d = AutoEncoder(cfg.MODEL_3D.FEAT_DIM, cfg.MODEL_2D.FEAT_DIM, cfg.AE.EMBED_DIM, cfg.AE.HIDDEN_LAYERS)

    return ae_2d_3d, ae_3d_2d

def build_encoder(cfg):
    import torch.nn as nn
    
    encoder_2d = nn.Sequential(*[nn.LeakyReLU() if i%2 else 
                                 nn.Linear(cfg.MODEL_2D.FEAT_DIM if i == 0 else cfg.AE.EMBED_DIM, cfg.AE.EMBED_DIM)
                                 for i in range(2*cfg.AE.HIDDEN_LAYERS + 3)])
    encoder_3d = nn.Sequential(*[nn.LeakyReLU() if i%2 else 
                                 nn.Linear(cfg.MODEL_3D.FEAT_DIM if i == 0 else cfg.AE.EMBED_DIM, cfg.AE.EMBED_DIM)
                                 for i in range(2*cfg.AE.HIDDEN_LAYERS + 3)])
    return encoder_2d, encoder_3d

def build_decoder(cfg):
    import torch.nn as nn
    
    decoder_2d = nn.Sequential(*[nn.LeakyReLU() if i%2 else 
                                 nn.Linear(cfg.AE.EMBED_DIM, cfg.MODEL_2D.FEAT_DIM if i == 2*cfg.AE.HIDDEN_LAYERS + 2 else cfg.AE.EMBED_DIM)
                                 for i in range(2*cfg.AE.HIDDEN_LAYERS + 3)])
    decoder_3d = nn.Sequential(*[nn.LeakyReLU() if i%2 else 
                                 nn.Linear(cfg.AE.EMBED_DIM, cfg.MODEL_3D.FEAT_DIM if i == 2*cfg.AE.HIDDEN_LAYERS + 2 else cfg.AE.EMBED_DIM)
                                 for i in range(2*cfg.AE.HIDDEN_LAYERS + 3)])
    return decoder_2d, decoder_3d
