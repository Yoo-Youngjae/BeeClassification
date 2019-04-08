import cv2 as cv
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

import cv2 as cv
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

def transBG2GW(imgSrc):
    img = cv.imread(imgSrc, cv.IMREAD_COLOR)

    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)



    # hist, bin = np.histogram(imgSrc.flatten(), 256, [0, 256])
    # cdf = hist.cumsum()
    # cdf_mask = np.ma.masked_equal(cdf, 0)
    # cdf_mask = np.uint8((cdf_mask - cdf_mask.min())*255/(cdf_mask.max()-cdf_mask.min()))
    # cdf = np.ma.filled(cdf_mask, 0).asype('uint8')
    # equ = cdf[gray]
    # res = np.hstck((equ))
    # cv.imshow('res',res)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    # 위의 것을 거친과정이 아래의 한줄이다.
    # histogram 평활화 하려면 아래걸로.
    equ_img = cv.equalizeHist(gray)

    # OTSH 기법 이용하려면 이걸로
    ret, th1 = cv.threshold(equ_img, 100, 255, cv.THRESH_OTSU)
    #
    # plt.imshow(th1, cmap='gray')
    # plt.show()


    # ret, th1 = cv.threshold(gray, 100, 255, cv.THRESH_BINARY)
    return gray

# mel_list = glob('mellifera/*.jpg')
# img = cv.imread(mel_list[10], cv.IMREAD_COLOR)
#
# gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
# ret, th1 = cv.threshold(gray, 100, 255, cv.THRESH_OTSU)
#
#
# image = np.array(th1)
# print(image.shape)
#
#
#
# plt.imshow(th1, cmap='gray')
# plt.show()