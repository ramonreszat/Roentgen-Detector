import numpy as np

from mxnet.gluon import nn
from mxnet import nd,gluon,autograd

from network.resnet import RoentgenResnet
from network.rpn import ProposalNetwork, ROIAlignmentLayer
from network.decoder import AnchorBoxDecoder


class RoentgenFasterRCNN(gluon.nn.HybridBlock):
	"""Gluon implementation of the Faster R-CNN two-stage detector.

    Parameters
    ----------
    num_classes : number of outputs
        for the Fast R-CNN classification task.
	rpn_head : proposal mode
		for alternating and class agnostic training.

    """
	def __init__(self, num_classes=2, iou_threshold=0.7, sizes=[0.25,0.15,0.05], ratios=[2,1,0.5], rpn_head=False):
		super(RoentgenFasterRCNN, self).__init__()
		self.rpn_head = rpn_head

		# backbone feature extraction network
		self.resnet = RoentgenResnet(64, conv_arch=[(2, 64), (2, 128), (2, 256), (2, 512)])
		# RPN head region proposal network 
		self.rpn = ProposalNetwork(512, num_anchors=9, anchor_points=(32,32))
		# anchor boxes of fixed size and ratio
		self.anchor = AnchorBoxDecoder(32, iou_threshold=iou_threshold, sizes=[0.25,0.15,0.05], ratios=[2,1,0.5])

		if not rpn_head:
			self.alignment = ROIAlignmentLayer((8,8), spatial_scale=0.03125)

			# Fast R-CNN detector
			self.fast_rcnn = nn.HybridSequential()
			self.fast_rcnn.add(nn.Dense(2048, flatten=False, activation='relu'))
			self.fast_rcnn.add(nn.Dense(1024, flatten=False, activation='relu'))

			self.rcnn_bbox_offset = nn.Dense(4, flatten=False)
			self.rcnn_detector = nn.Dense(num_classes, flatten=False)


	def hybrid_forward(self, F, X):
		feature_map = self.resnet(X)

		# ROI classification and regression
		rpn_cls_scores, rpn_bbox_pred = self.rpn(feature_map)
		# decode bounding boxes from confidence scores and offsets
		rpn_rois_pred, rpn_rois_scores = self.anchor(rpn_bbox_pred)

		if not self.rpn_head:
			regions = self.alignment(feature_map, rpn_rois_pred)
			# 5 proposals post nms -> flatten if only one is allowed
			roi_features = self.fast_rcnn(F.reshape(regions, shape=(0,5,-1)))

			bbox_pred = rpn_rois_pred + self.rcnn_bbox_offset(roi_features)
			class_pred = self.rcnn_detector(roi_features)
		else:
			# use RPN auxiliary head
			bbox_pred = rpn_rois_pred
			class_pred = rpn_rois_scores
		return bbox_pred, class_pred