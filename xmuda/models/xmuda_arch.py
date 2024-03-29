import torch
import torch.nn as nn

from xmuda.models.resnet34_unet import UNetResNet34
from xmuda.models.scn_unet import UNetSCN


class Net2DSeg(nn.Module):
    def __init__(self,
                 num_classes: int,
                 dual_head: bool,
                 backbone_2d,
                 backbone_2d_kwargs: dict,
                 full_img: bool = False,
                 ):
        super(Net2DSeg, self).__init__()

        # 2D image network
        if backbone_2d == 'UNetResNet34':
            self.net_2d = UNetResNet34(**backbone_2d_kwargs)
            feat_channels = 64
        else:
            raise NotImplementedError('2D backbone {} not supported'.format(backbone_2d))

        # segmentation head
        self.linear = nn.Linear(feat_channels, num_classes)

        # 2nd segmentation head
        self.dual_head = dual_head
        if dual_head:
            self.linear2 = nn.Linear(feat_channels, num_classes)

        # Full Image Return
        self.full_img = full_img 

    def forward(self, data_batch):
        # (batch_size, 3, H, W)
        img = data_batch['img']
        img_indices = data_batch['img_indices']

        # 2D network
        feats = self.net_2d(img)

        # 2D-3D feature lifting
        img_feats = []
        for i in range(feats.shape[0]):
            img_feats.append(feats.permute(0, 2, 3, 1)[i][img_indices[i][:, 0], img_indices[i][:, 1]])
        img_feats = torch.cat(img_feats, 0)

        preds = {'feats': img_feats}

        if self.full_img:
            full_logits = self.linear(feats.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            preds['full_logit'] = full_logits

            logits = [] 
            for i in range(feats.shape[0]):
                logits.append(full_logits.permute(0, 2, 3, 1)[i][img_indices[i][:, 0], img_indices[i][:, 1]])
            preds['seg_logit'] = torch.cat(logits, 0)

            if self.dual_head:
                full_logits2 = self.linear2(feats)
                preds['full_logit2'] = full_logits2
                
                logits = [] 
                for i in range(feats.shape[0]):
                    logits.append(full_logits.permute(0, 2, 3, 1)[i][img_indices[i][:, 0], img_indices[i][:, 1]])
                preds['seg_logit2'] = torch.cat(logits, 0)

        else:
            # linear
            preds['seg_logit'] = self.linear(img_feats)
            if self.dual_head:
                preds['seg_logit2'] = self.linear2(img_feats)
    
        return preds

    def classify_features(self, features):
        return self.linear(features)
    
    def classifier_parameters(self):
        return self.linear.parameters()

    def xmuda_classifier_parameters(self):
        if self.dual_head:
            return self.linear2.parameters()
        return None
    
    def feat_encoder_parameters(self):
        return self.net_2d.parameters()

class Net3DSeg(nn.Module):
    def __init__(self,
                 num_classes,
                 dual_head,
                 backbone_3d,
                 backbone_3d_kwargs,
                 ):
        super(Net3DSeg, self).__init__()

        # 3D network
        if backbone_3d == 'SCN':
            self.net_3d = UNetSCN(**backbone_3d_kwargs)
        else:
            raise NotImplementedError('3D backbone {} not supported'.format(backbone_3d))

        # segmentation head
        self.linear = nn.Linear(self.net_3d.out_channels, num_classes)

        # 2nd segmentation head
        self.dual_head = dual_head
        if dual_head:
            self.linear2 = nn.Linear(self.net_3d.out_channels, num_classes)

    def forward(self, data_batch):
        feats = self.net_3d(data_batch['x'])
        x = self.linear(feats)

        preds = {
            'feats': feats,
            'seg_logit': x,
        }

        if self.dual_head:
            preds['seg_logit2'] = self.linear2(feats)

        return preds

    def classify_features(self, features):
        return self.linear(features)
    
    def classifier_parameters(self):
        return self.linear.parameters()

    def xmuda_classifier_parameters(self):
        if self.dual_head:
            return self.linear2.parameters()
        return None
    
    def feat_encoder_parameters(self):
        return self.net_3d.parameters()
    
def test_Net2DSeg():
    # 2D
    batch_size = 2
    img_width = 400
    img_height = 225

    # 3D
    num_coords = 2000
    num_classes = 11

    # 2D
    img = torch.rand(batch_size, 3, img_height, img_width)
    u = torch.randint(high=img_height, size=(batch_size, num_coords // batch_size, 1))
    v = torch.randint(high=img_width, size=(batch_size, num_coords // batch_size, 1))
    img_indices = torch.cat([u, v], 2)

    # to cuda
    img = img.cuda()
    img_indices = img_indices.cuda()

    net_2d = Net2DSeg(num_classes,
                      backbone_2d='UNetResNet34',
                      backbone_2d_kwargs={},
                      dual_head=True)

    net_2d.cuda()
    out_dict = net_2d({
        'img': img,
        'img_indices': img_indices,
    })
    for k, v in out_dict.items():
        print('Net2DSeg:', k, v.shape)


def test_Net3DSeg():
    in_channels = 1
    num_coords = 2000
    full_scale = 4096
    num_seg_classes = 11

    coords = torch.randint(high=full_scale, size=(num_coords, 3))
    feats = torch.rand(num_coords, in_channels)

    feats = feats.cuda()

    net_3d = Net3DSeg(num_seg_classes,
                      dual_head=True,
                      backbone_3d='SCN',
                      backbone_3d_kwargs={'in_channels': in_channels})

    net_3d.cuda()
    out_dict = net_3d({
        'x': [coords, feats],
    })
    for k, v in out_dict.items():
        print('Net3DSeg:', k, v.shape)


if __name__ == '__main__':
    test_Net2DSeg()
    test_Net3DSeg()
