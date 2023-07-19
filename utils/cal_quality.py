"""
The quality of the current frame is evaluated in the following two aspects.
    1. Redundancy
    2. Blurring
"""
import numpy as np
import cv2

# redundancy
def get_MSE(prev_frame, frame) :
    diff = cv2.absdiff(prev_frame, frame)
    MSE = np.mean(np.square(diff))
    return MSE
    
# blurring
def get_FFT(frame, radius=60) :
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    H, W = gray_frame.shape   # H, W
    centerX, centerY = map(int, [W/2.0, H/2.0])
    # fft
    fft = np.fft.fft2(gray_frame)
    fftShift = np.fft.fftshift(fft)
    # remove Blur
    fftShift[centerY - radius : centerY + radius, centerX - radius : centerX + radius] = 0
    # ifft
    fftShift = np.fft.ifftshift(fftShift)
    reFrame = np.fft.ifft2(fftShift)
    # magnitude
    magnitude = 20*np.log(np.abs(reFrame))
    magMean = np.mean(magnitude)
    return magMean