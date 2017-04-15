import cv2
import numpy as np

# class PreProcessing:
size = (160, 80)


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

def random_brightness(img):
    # shifted = img + 1.0
    shifted = img
    # img_max_value = max(shifted.flatten())
    # max_coef = 2.0/img_max_value
    # min_coef = max_coef - 0.1
    # coef = np.random.uniform(min_coef, max_coef)
    # coef = 1.0
    # return shifted * coef - 1.0
    return shifted * 0.008

def process(img):
    # img = warp(img)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # img = cv2.Canny(img, 300, 300)

    # img = random_brightness(img)

    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    # equalize the histogram of the Y channel
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])

    # convert the YUV image back to RGB format
    img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    img = warp(img)
    kernel_size = 13
    img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # convert it to hsv
    # h, s, v = cv2.split(hsv)
    # v += 5
    # final_hsv = cv2.merge((h, s, v))
    # img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)



    # img = np.expand_dims(img, axis=4)
    return img