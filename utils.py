import numpy as np
import cv2 as cv

def rgb_resize(rgb):
	rgb = cv.resize(rgb, dsize=(224, 224), interpolation=cv.INTER_CUBIC)
	rgb = np.float32(rgb)
	image = np.expand_dims(rgb, axis = 0)
	return image

