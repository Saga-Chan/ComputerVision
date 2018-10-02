import cv2
from matplotlib import pyplot as plt

waves = cv2.imread('images/waves.jpg', cv2.IMREAD_GRAYSCALE)
histo_1 = cv2.calcHist([waves], None, None, [256], [0, 256])
plt.plot(histo_1)
plt.xlim([0, 256])
plt.show()

beach = cv2.imread('images/beach.jpg', cv2.IMREAD_GRAYSCALE)
histo_2 = cv2.calcHist([beach], None, None, [256], [0, 256])
plt.plot(histo_2)
plt.xlim([0, 256])
plt.show()

dog = cv2.imread('images/dog.jpg', cv2.IMREAD_GRAYSCALE)
histo_3 = cv2.calcHist([dog], None, None, [256], [0, 256])
plt.plot(histo_3)
plt.xlim([0, 256])
plt.show()

polar = cv2.imread('images/polar.jpg', cv2.IMREAD_GRAYSCALE)
histo_4 = cv2.calcHist([polar], None, None, [256], [0, 256])
plt.plot(histo_4)
plt.xlim([0, 256])
plt.show()

bear = cv2.imread('images/bear.jpg', cv2.IMREAD_GRAYSCALE)
histo_5 = cv2.calcHist([bear], None, None, [256], [0, 256])
plt.plot(histo_5)
plt.xlim([0, 256])
plt.show()

lake = cv2.imread('images/lake.jpg', cv2.IMREAD_GRAYSCALE)
histo_6 = cv2.calcHist([lake], None, None, [256], [0, 256])
plt.plot(histo_6)
plt.xlim([0, 256])
plt.show()

moose = cv2.imread('images/moose.jpg', cv2.IMREAD_GRAYSCALE)
histo_7 = cv2.calcHist([moose], None, None, [256], [0, 256])
plt.plot(histo_7)
plt.xlim([0, 256])
plt.show()

# # threshold option, where if something is maybe an 80% match, then we say it's a match.
# for i in range(7):
#     res = cv2.matchTemplate(waves, template[i], cv2.TM_CCOEFF_NORMED)
#     threshold = 0.8
# #    loc = np.where(res >= threshold)
#     if (res >= threshold):
#         print("picture number : %s" % i)
#         print(template[i])
#         print("matching : %s" % res, "/ 1 with the original template.")


cv2.waitKey(0)
cv2.destroyAllWindows()
