# imports
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import gridspec

# arr = np.array([[1.2, 2.3, 4.0], [1.2, 3.4, 5.2], [0.0, 1.0, 1.3], [0.0, 1.0, 2e-1]])
# iarr = np.array([1, 2, 3], np.uint8)

# print(arr)
# print(arr.dtype)
# print(arr.ndim)
# print(arr.shape)

# arr *= 2.5
# iarr *= 2.5
#
# print(arr)
# print(iarr)

# is_greater_one = (arr >= 1.)
# print(is_greater_one)

# print(arr[0,0]) # First row, first column
# print(arr[1]) # The whole second row
# print(arr[:,2]) # The third column

# print("Before: {}".format(arr[1,0]))
# view = arr[1]
# view[0] += 100
# print("After: {}".format(arr[1,0]))

# print(arr.mean())
# print(arr.mean(axis=0))

# is_greater_one = (arr > 1)
# print(is_greater_one.mean())

# load an image
# img = cv2.imread('images/cactus_5_ban.png')
# load an image as a single channel grayscale
# img_single_channel = cv2.imread('images/IMG_0005.png', 0)
# print some details about the images
# print('The shape of img without second arg is: {}'.format(img.shape))
# print('The shape of img_single_channel is: {}'.format(img_single_channel.shape))

# display the image with OpenCV imshow()
# cv2.imshow('OpenCV imshow()', img)
# OpenCV waitKey() is a required keyboard binding function after imwshow()
# cv2.waitKey(0)
# destroy all windows command
# cv2.destroyAllWindows()

# Saving an Image on a key press
# img = cv2.imread('images/cactus_5_ban.png')
# cv2.imshow('Option to Save image', img)
# print("press 's' to save the image as 'image_test_2.png\n")
# key = cv2.waitKey(0) # NOTE: if you are using a 64-bit machine, this needs to be: key = cv2.waitKey(0) & 0xFF
# if key == 27: # wait for the ESC key to exit
#     cv2.destroyAllWindows()
# elif key == ord('s'): # wait for 's' key to save and exit
#     cv2.imwrite('image_test_2.png', img)
#     cv2.destroyAllWindows()
# write an image with imwrite
# image_to_save = 'images/IMG_0194.JPG'
# cv2.imwrite(image_to_save, img)
# print('Image saved as {}'.format(image_to_save))

# # read and display the image as a reference
# img = cv2.imread('images/salade_de_fruits.jpg')
# cv2.imshow('Fruit Image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# height, width, channels = img.shape[:3]
# print('Image height: {}, Width: {}, # of channels: {}'.format(height, width, channels))
#
# blues = img[:, :, 0]
# greens = img[:, :, 1]
# reds = img[:, :, 2]

# cv2.imshow('Fruit Blues', blues)
# cv2.imshow('Fruit Greens', greens)
# cv2.imshow('Fruit Reds', reds)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # plot values for each color plane on a specific row
# fig = plt.figure(figsize=(10, 4))
# gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])

# # original image
# ax0 = plt.subplot(gs[0])
# ax0.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # need to convert BGR to RGB
# ax0.axhline(50, color='black')  # show the row being used
# ax0.axvline(100, color='k'), ax0.axvline(225, color='k')  # ref lines
# # image slice
# ax1 = plt.subplot(gs[1])
# ax1.plot(blues[49, :], color='blue')
# ax1.plot(greens[49, :], color='green')
# ax1.plot(reds[49, :], color='red')
# ax1.axvline(100, color='k', linewidth=2), ax1.axvline(225, color='k', linewidth=2)
# plt.suptitle('Examen des valeurs du plan de couleur pour une seule ligne')
# plt.show()

# cropped = img[300:400, 500:750]
# cv2.imshow('Cropped Image', cropped)
# print("press 's' to save the image as cropped_bicycle.png\n")
# key = cv2.waitKey(0)  # if you are using a 64-bit machine see below
# # the above line should be: key = cv2.waitKey(0) & 0xFF
# if key == 27:  # wait for the ESC key to exit
#     cv2.destroyAllWindows()
# elif key == ord('s'):  # wait for 's' key to save and exit
#     cv2.imwrite('images/cropped_bicycle.png', img)
#     cv2.destroyAllWindows()
# # get the size of the cropped image
# height, width = cropped.shape[:2]
# print('Cropped Width: {}px, Cropped Height: {}px'.format(width, height))

# x = np.uint8([250])
# y = np.uint8([10])
# print('Open CV Addition {}'.format(cv2.add(x, y)))
# # 250+10 = 260 => 255)
# print('')
# print('Numpy Addition {}\n'.format(x+y))
# # 250+10 = 260 % 256 = 4)

bicycle = cv2.imread('images/bicycle.png')
print(bicycle.shape[:2])
dolphin = cv2.imread('images/dolphin.jpg')
print(dolphin.shape[:2])

if bicycle.shape[:2] == dolphin.shape[:2]:
    sum_img = cv2.add(bicycle, dolphin)  # add images together
    cv2.imshow('Summed Images', sum_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    scaled_img = cv2.add(bicycle, 50)
    cv2.imshow('Scalar Addition on Bicycle Image', scaled_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if bicycle.shape[:2] == dolphin.shape[:2]:
    diff = cv2.absdiff(bicycle, dolphin)
    cv2.imshow('Subtracted Images', diff)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if bicycle.shape[:2] == dolphin.shape[:2]:
    average_img = bicycle / 2 + dolphin / 2
    alt_average_img = cv2.add(bicycle, dolphin) / 2
    cv2.imshow('Averaged Images', average_img)
    cv2.imshow('Alt. Averaged Images', alt_average_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

cap = cv2.VideoCapture(0)

while(1):
    # Take each frame
    _, frame = cap.read()

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame, frame, mask=mask)
    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    cv2.imshow('res', res)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
