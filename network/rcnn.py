import numpy as np

from mxnet.gluon import nn
from mxnet import nd,symbol,gluon,init

from network.resnet import RoentgenResnet
from network.rpn import ProposalNetwork, ProposalLayer, ROIAlignmentLayer


class RoentgenFasterRCNN(gluon.nn.HybridBlock):
	def __init__(self, num_classes=2, joint_training=True, Nt=0.7, sizes=[0.25,0.15,0.05], ratios=[2,1,0.5]):
		super(RoentgenFasterRCNN, self).__init__()
		self.joint_training = joint_training

		self.resnet = RoentgenResnet(64, conv_arch=[(2, 64), (2, 128), (2, 256), (2, 512)])

		self.rpn = ProposalNetwork(512, num_anchors=9, anchor_points=(32,32))
		self.proposals = ProposalLayer(sizes, ratios, Nt=0.7)

		if joint_training:
			self.alignment = ROIAlignmentLayer((8,8), spatial_scale=0.03125)
			self.fc = nn.HybridSequential()
			self.fc.add(nn.Dense(2048, flatten=False, activation='relu'))
			self.fc.add(nn.Dense(1024, flatten=False, activation='relu'))

			self.bbox_offset = nn.Dense(4, flatten=False)
			self.class_pred= nn.Dense(num_classes, flatten=False)


	def hybrid_forward(self, F, X):
		feature_map = self.resnet(X)

		cls_scores, bbox_pred = self.rpn(feature_map)
		rois = self.proposals(cls_scores, bbox_pred)

		if self.joint_training:
			regions = self.alignment(feature_map, rois)
			# 5 proposals post nms -> flatten if only one is allowed
			roi_features = self.fc(F.reshape(regions, shape=(0,5,-1)))

			bboxes = F.slice_axis(rois, axis=2, begin=1, end=None) + self.bbox_offset(roi_features)
			classes = self.class_pred(roi_features)
		else:
			bboxes = F.slice_axis(rois, axis=2, begin=1, end=None)
			classes = F.slice_axis(rois, axis=2, begin=0, end=1)
		return bboxes, classes