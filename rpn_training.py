import json
import argparse

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
parser.add_argument('--label_epsilon', type=float, default=1E-8)

cfg = parser.parse_args()

with open('roentgen-training-params.json', 'w') as config:
    json.dump(vars(cfg), config)


# parse and load the SIIM-ACR dataset
train_data = DICOMFolderDataset('siim-acr-data/train-pneumothorax/**/**/*.dcm', 'siim-acr-data/train-sample.csv')
train_loader = gluon.data.DataLoader(train_data, cfg.batch_size, shuffle=True, num_workers=4)

valid_data = DICOMFolderDataset('siim-acr-data/dev-pneumothorax/**/**/*.dcm', 'siim-acr-data/dev-sample.csv')
valid_loader = gluon.data.DataLoader(valid_data, cfg.batch_size, shuffle=False, num_workers=4)


# Region Proposal Network (RPN) auxilliary head
pneumothorax = RoentgenFasterRCNN(2, iou_threshold=0.7, iou_output=True, sizes=[0.25,0.15,0.05], ratios=[2,1,0.5], rpn_head=True)
pneumothorax.hybridize()

pos_weight = nd.array([cfg.beta],ctx=ctx)
box_weight = nd.array([cfg.gamma,cfg.gamma,cfg.gamma,cfg.gamma],ctx=ctx)

rpn_huber_loss = gluon.loss.HuberLoss(rho=cfg.rho, weight=5E+1)
# TODO: change to softmax output
binary_cross_entropy = gluon.loss.SigmoidBinaryCrossEntropyLoss(weight=2E+2)

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

                batch_size = data.shape[0]
                X = data.reshape((batch_size, 1, 1024, 1024))

                box_weight = nd.broadcast_to(box_weight.reshape(1,1,4,1,1), (batch_size,9,4,32,32))

                nd.waitall()

                with autograd.record():
                    # proposal generation
                    rpn_cls_scores, rpn_bbox_offsets, rpn_gt_offsets, attention_masks = pneumothorax(X, labels)

                    # label smoothing
                    attention_masks = (1-cfg.label_epsilon) * attention_masks

                    # multi-task loss
                    rpn_cls_loss = binary_cross_entropy(rpn_cls_scores, attention_masks, pos_weight)
                    rpn_reg_loss = rpn_huber_loss(rpn_bbox_offsets, rpn_gt_offsets, box_weight)

                    rpn_pred_loss = rpn_cls_loss/cfg.Ncls + cfg.balance*rpn_reg_loss/cfg.Nreg
                    
                rpn_pred_loss.backward()
                trainer.step(data.shape[0])

                cumulated_bce += rpn_cls_loss.mean().asscalar()
                cumulated_huber += rpn_reg_loss.mean().asscalar()
                cumulated_loss += rpn_pred_loss.mean().asscalar()

                pg.update()

            log.add_scalar(tag='rpn_pred_loss', value=(cumulated_loss/(len(train_data)/16)), global_step=epoch)
            log.add_scalar(tag='rpn_cls_loss', value=(cumulated_bce/(len(train_data)/16)), global_step=epoch)
            log.add_scalar(tag='rpn_bbox_loss', value=(cumulated_huber/(len(train_data)/16)), global_step=epoch)


            nd.waitall()

            with tqdm(total=int(len(valid_data)/16), desc='Validate Region Proposals (RPN): Epoch {}'.format(epoch)) as pg:
                tp = 0
                fp = 0
                fn = 0

                n = 0
                cumulated_error = 0

                for i, (data, labels) in enumerate(valid_loader):
                    data = data.as_in_context(ctx)
                    labels = labels.as_in_context(ctx)

                    batch_size = data.shape[0]
                    X = data.reshape((batch_size, 1, 1024, 1024))

                    ###move to AnchorBoxDecoder
                    p = nd.broadcast_to(anchor_points.reshape(1,1,2,32,32), (batch_size,9,2,32,32))
                    s = nd.broadcast_to(anchor_boxes.reshape(1,9,2,1,1),(batch_size,9,2,32,32)) # aw = ah = s*sqrt(r)
                    A = nd.concat(p,s,dim=2)

                    G = nd.broadcast_to(labels.reshape(batch_size,1,4,1,1),(batch_size,9,4,32,32))

                    IOU = box_iou(A,G)

                    #QUICK FIX: return this from anchor box decoder
                    # fg/bg threshold
                    p = IOU > cfg.iou_threshold
                        
                    # max IOU if there is no overlap > Nt
                    m = IOU.max(axis=(1,2,3))

                    m = nd.where(m<=cfg.iou_threshold,m,-1*m)
                    m = nd.where(m==0,m-1,m)

                    for i in range(batch_size):
                        p[i] = p[i] + (IOU[i]==m[i])

                    p_reg = nd.broadcast_to(p.reshape(batch_size,9,1,32,32), (batch_size,9,4,32,32))
                    ###

                    nd.waitall()


                    # predict regions of interest
                    s,b = pneumothorax(X)
                    s = nd.sigmoid(s)
                    b = b.reshape(1,9,4,32,32)

                    # apply offsets
                    B = A + b

                    # calculate the confusion matrix for the classifier
                    tp += nd.broadcast_logical_and(s > cfg.nms_threshold, p).sum().asscalar()
                    fp += nd.broadcast_logical_and(s > cfg.nms_threshold, nd.logical_not(p)).sum().asscalar()
                    fn += nd.broadcast_logical_and(s <= cfg.nms_threshold, p).sum().asscalar()

                    # mean absolute error for bounding box regression
                    Y = nd.multiply(p_reg,G)
                    Y_hat = nd.multiply(p_reg,B)

                    cumulated_error += nd.abs(Y - Y_hat).sum().asscalar()
                    n += (Y > 0).sum().asscalar()

                    pg.update()

                log.add_scalar(tag='rpn_precision', value=(tp/((tp+fp) if tp>0 else 1)), global_step=epoch)
                log.add_scalar(tag='rpn_recall', value=(tp/((tp+fn) if tp>0 else 1)), global_step=epoch)
                log.add_scalar(tag='rpn_mae', value=(cumulated_error/n if n>0 else 0), global_step=epoch)

            pneumothorax.export("roentgen-region-proposal", epoch=epoch)