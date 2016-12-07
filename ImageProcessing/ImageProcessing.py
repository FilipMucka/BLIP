import numpy as np
import cv2 
import time

def BGSubtraction(sPathSourceFolder, sPathDestinationFolder, frameCount):
	print ("START Background subtraction")
	startTime = time.time()

	nFrames = 1
	pic = cv2.imread(sPathSourceFolder + "/frame_%d.jpg" %nFrames)
	picGray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
	avg = np.float32(picGray)
	nFrames += 1

	while nFrames <= frameCount:
		pic = cv2.imread(sPathSourceFolder + "/frame_%d.jpg" %nFrames)
		picGray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
		nFrames += 1
		cv2.accumulateWeighted(picGray, avg, 1/nFrames)
		pass

	cv2.imwrite(sPathDestinationFolder + "/background.jpg", avg)

	nFrames = 1
	while nFrames <= frameCount:
		pic = cv2.imread(sPathSourceFolder + "/frame_%d.jpg" %nFrames)
		picGray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
		picGray = picGray - avg
		cv2.normalize(picGray, picGray, 0, 255, cv2.NORM_MINMAX)

		cv2.imwrite(sPathDestinationFolder + "/frame_%d.jpg" %nFrames, picGray)
		nFrames += 1
		pass

	print ("END Background subtraction")
	print (time.time() - startTime)

def ImageGradients(sPathSourceFolder, sPathDestinationFolder, frameCount):
	print ("START Image gradients with Laplacian Derivatives")
	startTime = time.time()

	nFrames = 1
	while nFrames <= frameCount:
		pic = cv2.imread(sPathSourceFolder + "/frame_%d.jpg" %nFrames)
		pic = cv2.Laplacian(pic, cv2.CV_64F)
		cv2.normalize(pic, pic, 0, 255, cv2.NORM_MINMAX)
		cv2.imwrite(sPathDestinationFolder + "/frame_%d.jpg" %nFrames, pic)
		nFrames += 1
		pass

	print ("END Image gradients with Laplacian Derivatives")
	print (time.time() - startTime)

def Threshold(sPathSourceFolder, sPathDestinationFolder, frameCount, boundaryValue):
	print ("START Thresholding")
	startTime = time.time()

	nFrames = 1
	while nFrames <= frameCount:
		pic = cv2.imread(sPathSourceFolder + "/frame_%d.jpg" %nFrames) 
		ret, pic = cv2.threshold(pic, boundaryValue, 255, cv2.THRESH_BINARY)
		cv2.imwrite(sPathDestinationFolder + "/frame_%d.jpg" %nFrames, pic)
		nFrames += 1
		pass

	print ("END Thresholding")
	print (time.time() - startTime)

BGSubtraction("SourceImages", "BGSubtraction", 120)
ImageGradients("BGSubtraction", "ImageGradients", 120)
Threshold("ImageGradients", "Threshold", 120, 105)
