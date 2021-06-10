from PIL import Image

class ResizeToFill(object):
	def __init__(self, width, height):
		self.width = width
		self.height =  height
		self.aspect_ratio = width / height

	def __call__(self, image):
		height, width = image.size
		aspect_ratio = width / height

		if aspect_ratio > self.aspect_ratio:
			# Crop left and right
			new_width = int(aspect_ratio * self.height)
			image = image.resize(
				(new_width, self.height),
				Image.BILINEAR
			)
			crop = int((new_width - self.width) / 2)
			image = image.crop((crop, 0, new_width - crop, self.height))

		else:
			new_height = int(self.width / aspect_ratio)
			image = image.resize(
				(self.width, new_height),
				Image.BILINEAR
			)
			crop = int((new_height - self.height) / 2)
			image = image.crop((0, crop, self.width, new_height - crop))
		

		image = image.resize((self.width, self.height), Image.BILINEAR) # Force correct size
		return image