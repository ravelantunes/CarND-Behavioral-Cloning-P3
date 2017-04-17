import cv2
import numpy as np

# Pre-processing size, this value is accessed by moth train_model.py and drive.py
size = (160, 80)

# Function to warp the image into a perspective from the top of the car looking down
def warp(img):
    img_size = (img.shape[1], img.shape[0])

    src = np.float32(
        [[100, 66],
         [220, 66],
         [320, 140],
         [0, 140]]
    )
    margin_y = 60
    dst = np.float32(
        [[margin_y, 0],
         [320 - margin_y, 0],
         [320 - margin_y, 160],
         [margin_y, 160]]
    )

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped

def process(img):
    # equalize the histogram of the Y channel
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    # convert the YUV image back to RGB format
    img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    # Remove perspective
    img = warp(img)

    # Gaussian blur
    kernel_size = 13
    img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    return img