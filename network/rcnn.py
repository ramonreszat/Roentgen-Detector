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
	def __init__(self, num_classes, iou_threshold=0.7, cls_threshold=0.5, sizes=[0.25,0.15,0.05], ratios=[2,1,0.5], rpn_head=False, iou_output=False):
		super(RoentgenFasterRCNN, self).__init__()
		self.rpn_head = rpn_head
		self.iou_output = iou_output

		# backbone feature extraction network
		self.resnet = RoentgenResnet(64, conv_arch=[(2, 64), (2, 128), (2, 256), (2, 512)])
		# RPN head region proposal network 
		self.rpn = ProposalNetwork(512, num_anchors=9, anchor_points=(32,32))
		# anchor boxes of fixed size and ratio
		self.anchor_decoder = AnchorBoxDecoder(32, iou_threshold=iou_threshold, cls_threshold=cls_threshold,
								sizes=[0.25,0.15,0.05], ratios=[2,1,0.5], rpn_head=rpn_head, iou_output=iou_output)

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

		if self.rpn_head:
			# split cls scores, needs to be rearranged for more classes
			rpn_cls_scores = F.reshape(rpn_cls_scores,(0,9,2,32,32))

			# decode offsets for training
			if self.iou_output:
				rpn_bbox_anchors, rpn_bbox_offsets, rpn_gt_offsets, rpn_ground_truth, attention_mask, rpn_bbox_ious = self.anchor_decoder(rpn_cls_scores, rpn_bbox_offsets, labels)
				return rpn_cls_scores, rpn_bbox_anchors, rpn_bbox_offsets, rpn_gt_offsets, rpn_ground_truth, attention_mask, rpn_bbox_ious
			else:
				rpn_bbox_anchors, rpn_bbox_offsets, rpn_gt_offsets, attention_mask = self.anchor_decoder(rpn_cls_scores, rpn_bbox_offsets, labels)
				return rpn_cls_scores, rpn_bbox_offsets, rpn_gt_offsets, attention_mask

		else:
			# TODO: bring ROIS into standard corner format
			rpn_bbox_rois = self.anchor_decoder(rpn_cls_scores, rpn_bbox_offsets)

			# TODO: non maximum suppression (NMS)

			regions = self.alignment(feature_map, rpn_bbox_rois)
			# 5 proposals post nms -> flatten if only one is allowed
			roi_features = self.fast_rcnn(F.reshape(regions, shape=(0,5,-1)))

			# TODO: add from a list for soft nms or watch for the shapes
			rcnn_bbox_pred = rpn_bbox_rois + self.rcnn_bbox_offset(roi_features)
			rcnn_class_pred = self.rcnn_detector(roi_features)

			return rcnn_bbox_pred, rcnn_class_pred