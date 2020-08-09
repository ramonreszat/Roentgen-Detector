from mxnet import gluon
from mxnet.gluon import nn

from network.rpn import RegionProposal
from network.resnet import ResnetBlock

class RoentgenFasterRCNN(gluon.nn.HybridBlock):
	def __init__(self):
		super(RoentgenFasterRCNN, self).__init__()

		self.encode = nn.HybridSequential(prefix='resnet_18')

		conv_arch = [(2, 64), (2, 128), (2, 256), (2, 512)]

		self.encode.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3))
		self.encode.add(nn.BatchNorm(), nn.Activation('relu'))
		self.encode.add(nn.MaxPool2D(pool_size=3, strides=2, padding=1))

		for i, (num_residuals, num_channels) in enumerate(conv_arch, start=1):
			if i==1:
				self.encode.add(ResnetBlock(num_residuals, num_channels, first_block=True))
			else:
				self.encode.add(ResnetBlock(num_residuals, num_channels))

		self.rpn = RegionProposal(512, num_anchors=9, anchor_points=(32,32))

	def hybrid_forward(self, F, X):
		x = self.encode(X)
		roi = self.rpn(x)
		return roi