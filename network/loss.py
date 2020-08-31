import numpy as np
from mxnet import nd,gluon
from mxnet import autograd

class RPNLoss(gluon.Block):
    def __init__(self, rpn_batch_size=16, bbox_weights=(1.0, 1.0, 1.0, 1.0), **kwargs):
        super(RPNLoss, self).__init__(**kwargs)
        self._rpn_batch_size = self.params.get_constant('rpn_batch_size', [rpn_batch_size])
        bbox_weights = np.array(bbox_weights).reshape((1, 4, 1))
        self._bbox_weights = self.params.get_constant('bbox_weights', bbox_weights)
    
    def forward(self, rpn_cls_prob, rpn_bbox_pred, rpn_cls_gt, rpn_bbox_gt):
        #with autograd.pause():
        ctx = rpn_cls_prob.context
        batch_size = rpn_cls_prob.shape[0]
            # construct cls_mask to ignore label=-1
        cls_mask = nd.stack(rpn_cls_gt == 0, rpn_cls_gt == 1, axis=1)
        bbox_weights = (rpn_cls_gt == 1).reshape(batch_size, 1, -1) * self._bbox_weights.data(ctx)
        
        # reshape -> (batch_size, 2, num_anchors*feat_h*feat_w)
        rpn_cls_log = nd.log(nd.clip(rpn_cls_prob.reshape((batch_size, 2, -1)), 1e-14, 1))
        cls_log_loss = - nd.sum(rpn_cls_log * cls_mask) / self._rpn_batch_size.data(ctx)

        # reshape -> (batch_size, 4, num_anchors*feat_h*feat_w)
        rpn_bbox_smooth_l1 = nd.smooth_l1(rpn_bbox_pred.reshape((batch_size, 4, -1)) - rpn_bbox_gt, scalar=3.0)
        bbox_smooth_l1_loss = nd.sum(rpn_bbox_smooth_l1 * bbox_weights) / self._rpn_batch_size.data(ctx)

        return cls_log_loss, bbox_smooth_l1_loss