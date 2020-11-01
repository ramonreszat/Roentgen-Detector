from mxnet import gluon
from mxnet.gluon import nn

class Residual(gluon.nn.HybridBlock):
	def __init__(self, num_channels, use_1x1conv=False, strides=1):
		super(Residual, self).__init__()

		self.conv1 = nn.Conv2D(num_channels, kernel_size=3, strides=strides, padding=1)
		self.bn1 = nn.BatchNorm()
		self.relu1 = nn.Activation('relu')
		self.conv2 = nn.Conv2D(num_channels, kernel_size=3, padding=1)
		self.bn2 = nn.BatchNorm()

		if use_1x1conv:
			self.conv3 = nn.Conv2D(num_channels, kernel_size=1, strides=strides)
		else:
			self.conv3 = None
		
		self.relu2 = nn.Activation('relu')

	def hybrid_forward(self, F, X):
		Y = self.relu1(self.bn1(self.conv1(X)))
		y = self.bn2(self.conv2(Y))

		if self.conv3:
			x = self.conv3(X)
		else:
			x = X
		return self.relu2(y + x)

def stack_resnet_block(num_residuals, num_channels, first_block=False):
	out = nn.HybridSequential()
	for i in range(num_residuals):
		if i==0 and not first_block:
			out.add(Residual(num_channels, use_1x1conv=True, strides=2))
		else:
			out.add(Residual(num_channels))
	return out