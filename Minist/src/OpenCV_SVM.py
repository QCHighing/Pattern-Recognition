import cv2 as cv
import numpy as np

SZ = 20
bin_n = 16  # Number of bins
affine_flags = cv.WARP_INVERSE_MAP | cv.INTER_LINEAR


# 抗扭斜处理
def deskew(img):
    m = cv.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11'] / m['mu02']
    M = np.float32([[1, skew, -0.5 * SZ * skew], [0, 1, 0]])
    img = cv.warpAffine(img, M, (SZ, SZ), flags=affine_flags)
    return img


# 计算图像的 HOG 描述符
def hog(img):
    gx = cv.Sobel(img, cv.CV_32F, 1, 0)
    gy = cv.Sobel(img, cv.CV_32F, 0, 1)
    mag, ang = cv.cartToPolar(gx, gy)
    bins = np.int32(bin_n * ang / (2 * np.pi))    # quantizing binvalues in (0...16)
    bin_cells = bins[:10, :10], bins[10:, :10], bins[:10, 10:], bins[10:, 10:]
    mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)     # hist is a 64 bit vector
    return hist


def main():
    # 读入数据集文件
    img = cv.imread('../img/digits.png', 0)
    if img is None:
        raise Exception("we need the digits.png image from samples/data here !")

    # 切割图像获得数据集
    cells = [np.hsplit(row, 100) for row in np.vsplit(img, 50)]  # 先上下分成50份，再左右分成100份

    # 划分训练集和测试集，各50%
    train_cells = [i[:50] for i in cells]
    test_cells = [i[50:] for i in cells]

    # 数据预处理
    deskewed = [list(map(deskew, row)) for row in train_cells]
    hogdata = [list(map(hog, row)) for row in deskewed]
    trainData = np.float32(hogdata).reshape(-1, 64)
    responses = np.repeat(np.arange(10), 250)[:, np.newaxis]

    # 创建向量机
    svm = cv.ml.SVM_create()
    svm.setKernel(cv.ml.SVM_LINEAR)
    svm.setType(cv.ml.SVM_C_SVC)
    svm.setC(2.67)
    svm.setGamma(5.383)

    # 开始训练
    svm.train(trainData, cv.ml.ROW_SAMPLE, responses)
    svm.save('svm_data.dat')

    # 处理测试数据
    deskewed = [list(map(deskew, row)) for row in test_cells]
    hogdata = [list(map(hog, row)) for row in deskewed]
    testData = np.float32(hogdata).reshape(-1, bin_n * 4)

    # 开始测试
    result = svm.predict(testData)[1]
    mask = result == responses
    correct = np.count_nonzero(mask)
    print(correct * 100.0 / result.size)


if __name__ == '__main__':
    main()
