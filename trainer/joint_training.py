import mxnet as mx
from mxnet import gluon,autograd

from data.dicom import DICOMFolderDataset
from network.rcnn import RoentgenFasterRCNN

from mxnet.symbol.contrib import box_iou
from mxnet.symbol import MakeLoss

from tqdm import tqdm
from mxboard import SummaryWriter

# devices
ctx = mx.gpu(0)

epochs = 100
batch_size=16


train_data = DICOMFolderDataset('siim-acr-data/train-pneumothorax/**/**/*.dcm', 'siim-acr-data/train.csv')
train_loader = gluon.data.DataLoader(train_data, 16, shuffle=True, num_workers=1)

valid_data = DICOMFolderDataset('siim-acr-data/dev-pneumothorax/**/**/*.dcm', 'siim-acr-data/train-dev.csv')
valid_loader = gluon.data.DataLoader(valid_data, 16, shuffle=False, num_workers=1)


pneumothorax = RoentgenFasterRCNN(num_classes=2, Nt=0.7, sizes=[0.25,0.15,0.05], ratios=[2,1,0.5])

# Kaiming initialization from uniform [-c,c] c = sqrt(2/Nin)
pneumothorax.collect_params().initialize(mx.init.Xavier(factor_type='in', magnitude=0.44444), ctx=ctx)
pneumothorax.hybridize()


trainer = gluon.Trainer(pneumothorax.collect_params(), 'adam', {'learning_rate': 3E-4, 'wd': 0.0001})

with SummaryWriter(logdir='./logs/pneumothorax-fasterrcnn') as log:
    for epoch in range(epochs):
        with tqdm(total=int(len(train_data)/batch_size), desc='Joint Training (RPN + Fast R-CNN): Epoch {}'.format(epoch)) as pg:

            for i, (data, labels) in enumerate(valid_loader):
                data = data.as_in_context(ctx)
                gtboxes = labels.as_in_context(ctx)

                X = data.reshape((0, 1, 1024, 1024))

                with autograd.record():
                    bboxes, classes = pneumothorax(X)

                    # binary classification
                    # nd box_iou
                    recognized = box_iou(bboxes, gtboxes, format='corner') > 0.3

                    cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
                    class_loss = MakeLoss(cross_entropy(classes,recognized), normalization='batch')

                    l1 = recognized * mx.sym.abs(bboxes - labels)
                    reg_loss = MakeLoss(l1, valid_thresh=1, normalization='valid')

                    detector_loss = class_loss + 10*reg_loss
                    L = MakeLoss(detector_loss, grad_scale=1.0, name='detection_loss')
                
                L.backward()
                trainer.step(data.shape[0])

                pg.update()
            
            log.add_scalar(tag='rcnn_loss', value=L.mean().asscalar(), global_step=epoch)
            log.add_scalar(tag='rcnn_bbox', value=reg_loss.mean().asscalar(), global_step=epoch)
            log.add_scalar(tag='rcnn_class', value=class_loss.mean().asscalar(), global_step=epoch)
