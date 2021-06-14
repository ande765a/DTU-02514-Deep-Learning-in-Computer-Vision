import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock2d(nn.Module):
	def __init__(self, channels, **kwargs):
		super(ResBlock2d, self).__init__()
		
		self.cnn1 = nn.Conv2d(
				in_channels=channels, 
				out_channels=channels,
				**kwargs
		)
		
		self.bn1 = nn.BatchNorm2d(num_features=channels)
		
		self.cnn2 = nn.Conv2d(
				in_channels=channels, 
				out_channels=channels,
				**kwargs
		)
		
		self.bn2 = nn.BatchNorm2d(num_features=channels)
		
	def forward(self, X):
		out = self.cnn1(X)
		out = self.bn1(out)
		out = F.relu(out)
		out = self.cnn2(out)
		out = self.bn2(out)
		out = X + out
		out = F.relu(out)
		return out

class ResConv2d(nn.Module):
	def __init__(self, in_channels, out_channels, **kwargs):
		super(ResConv2d, self).__init__()
		self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
		self.bn = nn.BatchNorm2d(num_features=out_channels)
		self.resblock = ResBlock2d(channels=out_channels, **kwargs)

	def forward(self, X):
		out = self.conv(X)
		out = self.bn(out)
		out = F.relu(out)
		out = self.resblock(out)
		return out

class DoubleConv2d(nn.Module):
	def __init__(self, in_channels, out_channels, **kwargs):
		super(DoubleConv2d, self).__init__()
		self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, **kwargs)
		self.bn1 = nn.BatchNorm2d(num_features=out_channels)

		self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, **kwargs)
		self.bn2 = nn.BatchNorm2d(num_features=out_channels)

	def forward(self, X):
		out = self.conv1(X)
		out = self.bn1(out)
		out = F.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = F.relu(out)

		return out

class Down(nn.Module):
	def __init__(self, in_channels, out_channels, **kwargs):
		super(Down, self).__init__()
		self.mp = nn.MaxPool2d(kernel_size=2)
		self.conv = DoubleConv2d(in_channels=in_channels, out_channels=out_channels, **kwargs) 

	def forward(self, X):
		out = self.mp(X)
		out = self.conv(out)
		return out

class Up(nn.Module):
	def __init__(self, in_channels, skip_channels, out_channels, **kwargs):
		super(Up, self).__init__()
		self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
		self.conv = DoubleConv2d(in_channels=in_channels + skip_channels, out_channels=out_channels, **kwargs)

	def forward(self, X, skip):
		out = self.upsample(X)
		out = torch.cat((out, skip), dim=1)
		out = self.conv(out)
		return out

class UNet(nn.Module):
	def __init__(self, in_channels, height, width):
		super(UNet, self).__init__()
		self.conv_in = DoubleConv2d(in_channels=in_channels, out_channels=64, kernel_size=1)
	
		# Downsample
		self.down1 = Down(in_channels=64, out_channels=128, kernel_size=3, padding=1)
		self.down2 = Down(in_channels=128, out_channels=256, kernel_size=3, padding=1)
		self.down3 = Down(in_channels=256, out_channels=512, kernel_size=3, padding=1)
		
		# Upsample
		self.up2 = Up(in_channels=512, skip_channels=256, out_channels=256, kernel_size=3, padding=1)
		self.up3 = Up(in_channels=256, skip_channels=128, out_channels=128, kernel_size=3, padding=1)
		self.up4 = Up(in_channels=128, skip_channels=64, out_channels=64, kernel_size=3, padding=1)

		self.conv_out = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1)
			
	def forward(self, X):
		skip1 = self.conv_in(X)
		skip2 = self.down1(skip1)
		skip3 = self.down2(skip2)
		latent = self.down3(skip3)

		# Upsample
		out = self.up2(latent, skip3)
		out = self.up3(out, skip2)
		out = self.up4(out, skip1)
		
		# Output
		mask_logits = self.conv_out(out)
		has_subtitles_conv = self.has_subtitle_conv(latent)

		mask = torch.sigmoid(mask_logits)
		
		return mask, mask_logits