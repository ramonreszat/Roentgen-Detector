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
	def __init__(self, num_classes, iou_threshold=0.7, iou_output=False, sizes=[0.25,0.15,0.05], ratios=[2,1,0.5], rpn_head=False):
		super(RoentgenFasterRCNN, self).__init__()
		self.rpn_head = rpn_head

		# backbone feature extraction network
		self.resnet = RoentgenResnet(64, conv_arch=[(2, 64), (2, 128), (2, 256), (2, 512)])
		# RPN head region proposal network 
		self.rpn = ProposalNetwork(512, num_anchors=9, anchor_points=(32,32))
		# anchor boxes of fixed size and ratio
		self.anchor = AnchorBoxDecoder(32, iou_threshold=iou_threshold, iou_output=iou_output, sizes=[0.25,0.15,0.05], ratios=[2,1,0.5])

		if not rpn_head:
			self.alignment = ROIAlignmentLayer((8,8), spatial_scale=0.03125)

			# Fast R-CNN detector
			self.fast_rcnn = nn.HybridSequential()
			self.fast_rcnn.add(nn.Dense(2048, flatten=False, activation='relu'))
			self.fast_rcnn.add(nn.Dense(1024, flatten=False, activation='relu'))

			self.rcnn_bbox_offset = nn.Dense(4, flatten=False)
			self.rcnn_detector = nn.Dense(num_classes, flatten=False)

	def hybrid_forward(self, F, X, labels=None):
		# extract features
		feature_map = self.resnet(X)
		
		# ROI classification and regression
		rpn_cls_scores, rpn_bbox_offsets = self.rpn(feature_map)

		if autograd.is_training and self.rpn_head:
			# decode offsets to prediction tensors
			gt_offsets, rpn_bbox_pred, attention_masks = self.anchor(rpn_bbox_offsets, labels)
			# split cls scores needs to be rearranged for more classes
			rpn_cls_scores = F.reshape(rpn_cls_scores,(0,9,2,32,32))
			
			return rpn_cls_scores, rpn_bbox_pred, gt_offsets, attention_masks

		elif self.rpn_head:
			# decode ROIs with TODO: IOU filtering
			rpn_bbox_pred, rpn_bbox_ious = self.anchor(rpn_bbox_offsets, labels)
			# split cls scores needs to be rearranged for more classes
			rpn_cls_scores = F.reshape(rpn_cls_scores,(0,9,2,32,32))

			return rpn_cls_scores, rpn_bbox_pred, rpn_bbox_ious

		else:
			# decode ROIs from offset prediction TODO: bring into standard corner format and NMS filter
			rpn_rois_pred = self.anchor(rpn_bbox_offsets)

			regions = self.alignment(feature_map, rpn_rois_pred)
			# 5 proposals post nms -> flatten if only one is allowed
			roi_features = self.fast_rcnn(F.reshape(regions, shape=(0,5,-1)))

			rcnn_bbox_pred = rpn_rois_pred + self.rcnn_bbox_offset(roi_features)
			rcnn_class_pred = self.rcnn_detector(roi_features)

			return rcnn_bbox_pred, rcnn_class_pred