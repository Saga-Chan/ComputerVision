# imports
import cv2

# load an image
flag = cv2.imread('images/gay_pride_flag.png')
# load an image as a single channel grayscale
flag_bw = cv2.imread('images/gay_pride_flag.png', cv2.IMREAD_GRAYSCALE)

flag_red_tons = flag.copy()  # Make a copy
flag_red_tons[:, :, 0] = 0
flag_red_tons[:, :, 1] = 0

flag_green_tons = flag.copy()  # Make a copy
flag_green_tons[:, :, 0] = 0
flag_green_tons[:, :, 2] = 0

flag_blue_tons = flag.copy()  # Make a copy
flag_blue_tons[:, :, 1] = 0
flag_blue_tons[:, :, 2] = 0

# display the image with OpenCV imshow()
cv2.imshow('Original Flag', flag)
cv2.imshow('Flag in Black & White', flag_bw)
cv2.imshow('Reds in Flag', flag_red_tons)
cv2.imshow('Greens in Flag', flag_green_tons)
cv2.imshow('Blues in Flag', flag_blue_tons)
# OpenCV waitKey() is a required keyboard binding function after imshow()
cv2.waitKey(0)
# destroy all windows command
cv2.destroyAllWindows()
