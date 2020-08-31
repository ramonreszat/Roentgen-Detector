import mxnet as mx
from mxnet import nd,gluon,autograd

from data.dicom import DICOMFolderDataset
from network.rcnn import RoentgenFasterRCNN
from network.loss import RPNLoss

from tqdm import tqdm

epochs = 100

# devices
ctx = mx.gpu(0)

train_data = DICOMFolderDataset('siim-acr-data/train-pneumothorax/**/**/*.dcm', 'siim-acr-data/train.csv')
train_loader = gluon.data.DataLoader(train_data, 16, shuffle=True, num_workers=1)

valid_data = DICOMFolderDataset('siim-acr-data/dev-pneumothorax/**/**/*.dcm', 'siim-acr-data/train-dev.csv')
valid_loader = gluon.data.DataLoader(valid_data, 16, shuffle=False, num_workers=1)

pneumothorax = RoentgenFasterRCNN(num_classes=2, rpn_output=True, Nt=0.7, scales=[0.25,0.15,0.05], ratios=[2,1,0.5])
pneumothorax.hybridize()

# Kaiming initialization from uniform [-c,c] c = sqrt(2/Nin)
pneumothorax.collect_params().initialize(mx.init.Xavier(factor_type='in', magnitude=0.44444), ctx=ctx)


trainer = gluon.Trainer(pneumothorax.collect_params(), 'adam', {'learning_rate': 1E-5, 'wd': 0.00001})

for epoch in range(epochs):
  batch_size = 16
  cumulated_cls_loss = 0.0
  cumulated_bbox_loss = 0.0

  with tqdm(total=int(len(valid_data)/batch_size), desc='Training Region Proposal Network: Epoch {}'.format(epoch)) as pg:

    for i, (data, labels) in enumerate(valid_loader):
      data = data.as_in_context(ctx)
      gtbox = labels.as_in_context(ctx)

      batch_size = data.shape[0]

      im_info = nd.array([[1024,1024,1]], ctx=mx.gpu(0))
      im_info = im_info.repeat(batch_size, axis=0)

      X = data.reshape((0, 1, 1024, 1024))
      gtbox = gtbox.reshape(0,1,4)

      rpn_loss = RPNLoss(rpn_batch_size=batch_size)
      rpn_loss.initialize(ctx=ctx)

      with autograd.record():
        rpn_cls_prob, rpn_bbox_pred, rpn_label, rpn_bbox_target = pneumothorax(X, im_info, gtbox)

        rpn_cls_loss, rpn_bbox_loss = rpn_loss(rpn_cls_prob, rpn_bbox_pred, rpn_label, rpn_bbox_target)
        rpn_loss = rpn_cls_loss + 10 * rpn_bbox_loss/1024

      autograd.backward(rpn_loss)
      trainer.step(data.shape[0])

      cumulated_cls_loss += rpn_cls_loss.asscalar()
      cumulated_bbox_loss += rpn_bbox_loss.asscalar()

      pg.update()
    
    print("Status: {} LOSS, {} CLS, {} BBOX".format(cumulated_cls_loss/66+cumulated_bbox_loss/66,cumulated_cls_loss/66,cumulated_bbox_loss/66))