from typing import List

import pytorch_lightning as pl
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchmetrics import JaccardIndex, F1Score
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.segmentation.deeplabv3 import ASPP

from losses import DiceLossMod, FocalLossMod


###############################################
###############################################
class ResNetASPP(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.save_hyperparameters()

        if kwargs['pretrained'] == True:
            weights=ResNet18_Weights.IMAGENET1K_V1
        else:
            weights=None

        return_layers = {"layer2": "out"}
        self.encoder = resnet18(weights=weights)
        self.encoder = IntermediateLayerGetter(self.encoder, return_layers=return_layers)

        if self.hparams.frozen_start:
            for param in self.encoder.parameters():
                param.requires_grad = False
    
        # create the head:
        self.decoder = nn.Sequential(ASPP(in_channels=128, atrous_rates = [12, 24, 36], out_channels = 128),
                                        nn.Conv2d(128, 128, 3, padding=1, bias=False),
                                        nn.BatchNorm2d(128),
                                        nn.ReLU(),
                                        nn.Conv2d(128, self.hparams['num_classes'], 1)
                                        )

        # ASPP has one global average pooling that messes things up 
        # in case we want to change the input size (full raster prediction)
        avgpool_replacer = nn.AvgPool2d(8,8)
        if isinstance(self.decoder[0].convs[-1][0], nn.AdaptiveAvgPool2d):
            self.decoder[0].convs[-1][0] = avgpool_replacer
        else:
            print('Check the model! Is there an AdaptiveAvgPool2d somewhere?')

        # initialize random weights with kaiming normal rather than kaiming uniform
        # for m in self.decoder.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        #     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

        # metrics
        # use the num_classes+1 to ignore last index (torchmetrics F1 does not allow arbitrary "ignore_index")
        # torchmetrics Jaccard Index does not work when average='weighted' & ignore_index is present
        # use macro as underperforming classes will have a higher influence in the final value
        self.jaccard = JaccardIndex(num_classes=self.hparams.num_classes+1, 
                                    average='macro', 
                                    ignore_index = self.hparams.num_classes)
        self.f1 = F1Score(num_classes=self.hparams.num_classes+1, 
                                    average='macro', 
                                    ignore_index = self.hparams.num_classes)

        # loss
        if self.hparams.loss == 'cross_entropy':
            self.loss_fn = nn.CrossEntropyLoss(ignore_index = self.hparams.num_classes)
        elif self.hparams.loss == 'focal':
            self.loss_fn = FocalLossMod(gamma=self.hparams.gamma, 
                                        alpha=self.hparams.alpha, 
                                        reduction='mean', 
                                        ignore_index = self.hparams.num_classes)

        elif self.hparams.loss == 'dice':
            self.loss_fn = DiceLossMod(ignore_index = self.hparams.num_classes)

        
    def forward(self, x:Tensor) -> Tensor:
        
        input_shape = x.shape[-2:]

        features = self.encoder(x)['out']

        logits = self.decoder(features)         
        logits = F.interpolate(logits, size=input_shape, mode="bilinear", align_corners=False)

        return logits
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                                    min_lr=1e-8,
                                                                    patience=self.hparams.reduce_lr_patience, 
                                                                    verbose=True)         
        return optimizer 
    
    def training_step(self, batch, batch_idx):

        x, y = batch
        # torchmetrics F1 does not allow arbitrary "ignore_index"
        y[y==self.hparams.ignore_index] = self.hparams.num_classes

        y_pred = self.forward(x)
        loss = self.loss_fn(y_pred, y)
        self.log('train_loss', loss, on_epoch=True)

        ###############################################
        # metrics
        ###############################################
        # IoU/Jaccard index
        iou = self.jaccard(torch.argmax(y_pred, axis=1), y)
        self.log('train_IoU', iou, on_epoch=True)

        # F1 
        f1 = self.f1(torch.argmax(y_pred, axis=1).ravel(), y.ravel())
        self.log('train_f1', f1, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        y[y==self.hparams.ignore_index] = self.hparams.num_classes

        y_pred = self.forward(x)
        loss = self.loss_fn(y_pred, y)
        self.log('val_loss', loss, on_epoch=True)

        ###############################################
        # metrics
        ###############################################
        # IoU/Jaccard index
        iou = self.jaccard(torch.argmax(y_pred, axis=1), y)
        self.log('val_IoU', iou, on_epoch=True)

        # F1
        f1 = self.f1(torch.argmax(y_pred, axis=1).ravel(), y.ravel())
        self.log('val_f1', f1, on_epoch=True)

        return loss

    def training_epoch_end(self, outputs):
        sch = self.scheduler

        # If the selected scheduler is a ReduceLROnPlateau scheduler.
        if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sch.step(self.trainer.callback_metrics["val_loss"])   

            if self.hparams.frozen_start and (sch.optimizer.param_groups[0]['lr'] < self.hparams.lr):
                for param in self.encoder.parameters():
                    param.requires_grad = True                    
