import torch
import torch.nn as nn

class auto_shallow(nn.Module):
    def __init__(self, n_class,seed=7):
        super(auto_shallow, self).__init__()
        torch.manual_seed(seed)


        self.enc_feat = nn.Sequential()
        self.enc_feat.add_module('conv1',nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,padding=1,stride=2))
        self.enc_feat.add_module('batch_norm1',nn.BatchNorm2d(64))
        self.enc_feat.add_module('relu1', nn.ReLU(True))
        self.enc_feat.add_module('conv2',nn.Conv2d(in_channels=64,out_channels=32,kernel_size=3,padding=1,stride=2))
        self.enc_feat.add_module('batch_norm2',nn.BatchNorm2d(32))
        self.enc_feat.add_module('relu2', nn.ReLU(True))
        self.enc_feat.add_module('conv3',nn.Conv2d(in_channels=32,out_channels=16,kernel_size=3,padding=1))
        self.enc_feat.add_module('batch_norm3',nn.BatchNorm2d(16))
        self.enc_feat.add_module('relu3', nn.ReLU(True))

        # decoder
        self.rec_feat = nn.Sequential()
        self.rec_feat.add_module('convt1',nn.ConvTranspose2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),output_padding=1))
        self.rec_feat.add_module('batch_normt1',nn.BatchNorm2d(32))
        self.rec_feat.add_module('relut1',nn.ReLU(True))
        self.rec_feat.add_module('convt2',nn.ConvTranspose2d(in_channels=32,out_channels=64,kernel_size=3,padding=1,stride=2,output_padding=1))
        self.rec_feat.add_module('batch_normt2',nn.BatchNorm2d(64))
        self.rec_feat.add_module('relut2',nn.ReLU(True))
        self.rec_feat.add_module('convt3',nn.ConvTranspose2d(in_channels=64,out_channels=3,kernel_size=3,padding=1))
        self.rec_feat.add_module('batch_normt3',nn.BatchNorm2d(3))

        """
        self.rec_feat = nn.Sequential(
        nn.ConvTranspose2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),output_padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(True),
        nn.ConvTranspose2d(in_channels=32,out_channels=64,kernel_size=3,padding=1,stride=2,output_padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(True),
        nn.ConvTranspose2d(in_channels=64,out_channels=3,kernel_size=3,padding=1),
        nn.BatchNorm2d(3)
        )
        """

    def forward(self, input_data,embeddings=None):
        if embeddings is None:
            feat_code = self.enc_feat(input_data)
            feat_pred = feat_code.view(-1, 16 * 8 * 8)
            pred_label = None
        else:
            feat_code = embeddings.view(-1,16,8,8)
            feat_pred = None
            pred_label = None
        img_rec = self.rec_feat(feat_code)
        return pred_label,img_rec,feat_pred
