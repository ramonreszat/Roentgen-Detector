from mxnet import gluon
from mxnet.gluon import nn

from rcnn.residual import stack_resnet_block

class RoentgenResnet(gluon.nn.HybridBlock):
	"""Feature extraction using standard resnet architecture.

    Parameters
    ----------
    conv_arch : number of residuals and channels
        with 1x1conv at each of the first blocks and increasing feature channels.

    """
	def __init__(self,num_channels, conv_arch=[(2, 64), (2, 128), (2, 256), (2, 512)]):
		super(RoentgenResnet, self).__init__()

		self.conv0 = nn.Conv2D(num_channels, kernel_size=7, strides=2, padding=3)
		self.bn0 = nn.BatchNorm()
		self.relu0 = nn.Activation('relu')
		self.pool0 = nn.MaxPool2D(pool_size=3, strides=2, padding=1)

		self.residuals = nn.HybridSequential()

		for i, (num_residuals, num_channels) in enumerate(conv_arch):
			if i==0:
				self.residuals.add(stack_resnet_block(num_residuals, num_channels, first_block=True))
			else:
				self.residuals.add(stack_resnet_block(num_residuals, num_channels))
	
	def hybrid_forward(self, F, X):
		Y = self.relu0(self.bn0(self.conv0(X)))
		x = self.residuals(self.pool0(Y))
		return x