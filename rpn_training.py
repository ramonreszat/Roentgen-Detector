import json
import argparse

import pandas as pd
from mxnet import nd,cpu,gpu,init,gluon,autograd

from data.dicom import DICOMFolderDataset
from network.rcnn import RoentgenFasterRCNN
from network.decoder import AnchorBoxDecoder

from tqdm import tqdm
from mxboard import SummaryWriter

# devices
ctx = cpu(0)

parser = argparse.ArgumentParser()

# configure training parameter
parser.add_argument('--epochs', '-epochs', type=int)
parser.add_argument('--batch_size', '-batch', type=int, default=16)
parser.add_argument('--learning_rate', '-lr', type=float, default=1E-5)
parser.add_argument('--weight_decay', '-wd', type=float, default=1E-6)
parser.add_argument('--beta', '-beta', type=float, default=1.7, help='Value > 1 increases recall for the RPN')
parser.add_argument('--gamma', '-gamma', type=float, default=1.5)
parser.add_argument('--rho', '-rho', type=float, default=1)
parser.add_argument('--balance', '-lambda', type=int, default=10, help='Balance between classification and regression loss')
parser.add_argument('--Ncls', '-Ncls', type=int, default=16)
parser.add_argument('--Nreg', '-Nreg', type=int, default=1024)
parser.add_argument('--nms_threshold', '-nms', type=float, default=0.5)
parser.add_argument('--iou_threshold', type=float, default=0.7, help='Threshold for an anchor box to be picked as forground')

cfg = parser.parse_args()

df_rpn_train = pd.DataFrame()
df_rpn_valid = pd.DataFrame()

with open('roentgen-training-params.json', 'w') as config:
    json.dump(vars(cfg), config)


# parse and load the SIIM-ACR dataset
train_data = DICOMFolderDataset('siim-acr-data/train-pneumothorax/**/**/*.dcm', 'siim-acr-data/train-sample.csv')
train_loader = gluon.data.DataLoader(train_data, cfg.batch_size, shuffle=True, num_workers=4)

valid_data = DICOMFolderDataset('siim-acr-data/dev-pneumothorax/**/**/*.dcm', 'siim-acr-data/dev-sample.csv')
valid_loader = gluon.data.DataLoader(valid_data, cfg.batch_size, shuffle=False, num_workers=4)


# Region Proposal Network (RPN) auxilliary head
pneumothorax = RoentgenFasterRCNN(2, iou_threshold=0.7, iou_output=True, sizes=[0.25,0.15,0.05], ratios=[2,1,0.5], rpn_head=True)
pneumothorax.hybridize(active=False)

pos_weight = nd.array([cfg.beta],ctx=ctx)
box_weight = nd.array([cfg.gamma,cfg.gamma,cfg.gamma,cfg.gamma],ctx=ctx)

rpn_huber_loss = gluon.loss.HuberLoss(rho=cfg.rho, weight=5E+1)
rpn_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss(axis=2, weight=2E+2)

# Kaiming initialization from uniform [-c,c] c = sqrt(2/Nin) 
pneumothorax.collect_params().initialize(init.Xavier(factor_type='in', magnitude=0.44444), ctx=ctx)

trainer = gluon.Trainer(pneumothorax.collect_params(), 'adam', {'learning_rate': cfg.learning_rate, 'wd': cfg.weight_decay})

with SummaryWriter(logdir='./logs/pneumothorax-rpn') as log:
    for epoch in range(cfg.epochs):
        with tqdm(total=int(len(train_data)/cfg.batch_size), desc='Training Region Proposals (RPN): Epoch {}'.format(epoch)) as pg:
            cumulated_loss = 0.0
            cumulated_bce = 0.0
            cumulated_huber = 0.0

            for i, (data, labels) in enumerate(train_loader):
                data = data.as_in_context(ctx)
                labels = labels.as_in_context(ctx)

                data.attach_grad()
                batch_size = data.shape[0]
                X = data.reshape((batch_size, 1, 1024, 1024))

                box_weight = nd.broadcast_to(box_weight.reshape(1,1,4,1,1), (batch_size,9,4,32,32))

                nd.waitall()

                with autograd.record():
                    # proposal generation
                    rpn_cls_scores, rpn_bbox_offsets, rpn_gt_offsets, attention_masks = pneumothorax(X, labels)

                    # multi-task loss
                    print("object prediction mask: %s" % (rpn_cls_scores.shape,))
                    print("valid object mask: %s" % (attention_masks.shape,))
                    rpn_cls_loss = rpn_cross_entropy(rpn_cls_scores, attention_masks, pos_weight)
                    rpn_reg_loss = rpn_huber_loss(rpn_bbox_offsets, rpn_gt_offsets, box_weight)

                    rpn_pred_loss = rpn_cls_loss/cfg.Ncls + cfg.balance*rpn_reg_loss/cfg.Nreg
                    
                rpn_pred_loss.backward()
                trainer.step(data.shape[0])

                print("rpn cls loss: %s" % (rpn_cls_loss.shape,))
                print("rpn reg loss: %s" % (rpn_reg_loss.shape,))
                print("rpn pred loss: %s" % (rpn_pred_loss.shape,))
                print(rpn_reg_loss.mean())
                print(rpn_cls_loss.mean())
                print(rpn_pred_loss.mean())
                cumulated_bce += rpn_cls_loss.mean().asscalar()
                cumulated_huber += rpn_reg_loss.mean().asscalar()
                cumulated_loss += rpn_pred_loss.mean().asscalar()

                pg.update()
            
            # training metrics
            rpn_epoch_loss = {
                'rpn_pred_loss': cumulated_loss/(len(train_data)/cfg.batch_size),
                'rpn_cls_loss': cumulated_bce/(len(train_data)/cfg.batch_size),
                'rpn_reg_loss': cumulated_huber/(len(train_data)/cfg.batch_size)
            }
            df_rpn_train = df_rpn_train.append(rpn_epoch_loss, ignore_index=True)

            log.add_scalar(tag='rpn_pred_loss', value=rpn_epoch_loss.get('rpn_pred_loss'), global_step=epoch)
            log.add_scalar(tag='rpn_cls_loss', value=rpn_epoch_loss.get('rpn_cls_loss'), global_step=epoch)
            log.add_scalar(tag='rpn_reg_loss', value=rpn_epoch_loss.get('rpn_reg_loss'), global_step=epoch)


            nd.waitall()


            with tqdm(total=int(len(valid_data)/16), desc='Validate Region Proposals (RPN): Epoch {}'.format(epoch)) as pg:
                tp = 0
                fp = 0
                tn = 0
                fn = 0

                n = 0
                cumulated_error = 0

                for i, (data, labels) in enumerate(valid_loader):
                    data = data.as_in_context(ctx)
                    labels = labels.as_in_context(ctx)

                    batch_size = data.shape[0]
                    X = data.reshape((batch_size, 1, 1024, 1024))
                    
                    # feed forward to get evaluation tensors
                    rpn_bbox_pred, rpn_bbox_ious = pneumothorax(X, labels)
                    rpn_bbox_rois = nd.broadcast_to(labels.reshape(batch_size,1,4,1,1),(batch_size,9,4,32,32))

                    # count valid classifications
                    valid = rpn_bbox_ious > cfg.iou_threshold

                    nd.waitall()

                    # calculate the confusion matrix for the classifier
                    tp += nd.broadcast_logical_and(rpn_cls_scores > cfg.nms_threshold, valid).sum().asscalar()
                    fp += nd.broadcast_logical_and(rpn_cls_scores > cfg.nms_threshold, nd.logical_not(valid)).sum().asscalar()
                    tn += nd.broadcast_logical_and(rpn_cls_scores <= cfg.nms_threshold, nd.logical_not(valid)).sum().asscalar()
                    fn += nd.broadcast_logical_and(rpn_cls_scores <= cfg.nms_threshold, valid).sum().asscalar()

                    
                    Y = nd.multiply(valid, rpn_bbox_rois)
                    Y_hat = nd.multiply(valid, rpn_bbox_pred)

                    # valid normalized mean squared error for bounding box regression
                    cumulated_error += nd.square(Y - Y_hat).sum().asscalar()
                    n += (Y > 0).sum().asscalar()

                    pg.update()
                
                # validation metrics
                rpn_valid_metrics = {
                    'true_positive_rate': tp/(tp+fp) if tp+fp>0 else 0.0,
                    'false_positive_rate': fp/(tp+fp) if tp+fp>0 else 0.0,
                    'true_negative_rate': tn/(tn+fn) if tn+fn>0 else 0.0,
                    'false_negative_rate': fn/(tn+fn) if tn+fn>0 else 0.0,
                    'rpn_local_recall': tp/((tp+fn) if tp>0 else 1),
                    'rpn_local_precision': tp/((tp+fp) if tp>0 else 1.0),
                    'rpn_mean_squared_error': cumulated_error/n if n>0 else 0
                }
                df_rpn_valid = df_rpn_valid.append(rpn_valid_metrics, ignore_index=True)

                log.add_scalar(tag='rpn_local_recall', value=rpn_valid_metrics.get('rpn_local_recall'), global_step=epoch)
                log.add_scalar(tag='rpn_local_precision', value=rpn_valid_metrics.get('rpn_local_precision'), global_step=epoch)
                log.add_scalar(tag='rpn_mean_square_error', value=rpn_valid_metrics.get('rpn_mean_squared_error'), global_step=epoch)

            pneumothorax.export("roentgen-region-proposal-network", epoch=epoch)

df_rpn_train.to_csv('roentgen-training-loss.csv')
df_rpn_valid.to_csv('roentgen-training-metrics.csv')