import cv2
import numpy as np
from sklearn.cluster import DBSCAN

def contrast_img(image, a, b):

   
    """增强图像对比度。

        通过线性变换增强像素值之间的差距来增强对比度。

        Parameters
        ----------
        image : numpy.array
            RGB图像
        a : int
            比例系数
        b : int 
            常数

        Returns
        -------
        numpy.array
            增强对比度后的图像

        """

    O = float(a) * image + float(b)
    O[O>255] = 255 
    O[O<0] = 0
    O = np.round(O)
    O = O.astype(np.uint8)
    return O

def edgeWithSobel(image):

    """sobel边缘检测算法，opencv实现。

        sobel边缘检测算法的原理是通过计算灰度值的一阶导数，并找到极大极小值来确定边缘。\n
        这个方法在进行边缘检测前，增强了图片对比度。

        Parameters
        ----------
        image : numpy.array
            RGB图像

        Returns
        -------
        list
            [absX, absY, absXY] \n
            absX (numpy.array): 以X轴为方向卷积得到的边缘灰度图\n
            absY (numpy.array): 以Y轴为方向卷积得到的边缘灰度图\n
            absXY (numpy.array): 综合absX和absY得到的边缘灰度图\n

        """
    
    image_grey = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    for i in range(0,2):
        image_grey = contrast_img(image_grey, 3, -220)   #### enhance contrast

    x = cv2.Sobel(image_grey, cv2.CV_16S, 1, 0, None, 3) # 3 -> shape of sobel operator 
    y = cv2.Sobel(image_grey, cv2.CV_16S, 0, 1, None, 3)
    absX = cv2.convertScaleAbs(x)  
    absY = cv2.convertScaleAbs(y)
    absXY = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

    return absX, absY, absXY

def edgeWithCanny(image):

    """canny边缘检测算法，opencv实现。

        canny边缘检测算法的步骤：\n
        1.对输入图像进行高斯平滑。\n
        2.计算梯度幅度和方向来估计每一点处的边缘强度与方向。\n
        3.根据梯度方向，对梯度幅值进行非极大值抑制。\n
        4.用双阈值处理和连接边缘。\n
    
        这个方法在进行边缘检测前，增强了图片对比度。

        Parameters
        ----------
        image : numpy.array
            RGB图像

        Returns
        -------
        numpy.array
            边缘检测结果的灰度图

        """

    dst = contrast_img(image, 1.5)
    edges = cv2.Canny(dst,100,220)
    return edges

def dbscan(image, eps, min_samples):

    """dbscan聚类算法，skicit-learn实现。

        基于密度的聚类算法，具有不受点集合形状影响的特点。

        Parameters
        ----------
        image : numpy.array
            灰度图像
        eps : int
            扫描半径
        min_samples : int 
            最小包含点数

        Returns
        -------
        list
            [Data, n_cluster, labels]\n
            Data (numpy.array): 输入的灰度图提取出来的点集合\n
            n_cluster (int): 类别数目\n
            labels (list): 类别标签\n

        """

    Data = np.where(image!=0)
    Data = np.asarray(Data)
    if(Data.shape[0]==0 or Data.shape[1]==0):
        return None, 0, None
    Data = Data.T
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(Data)
    labels = db.labels_
    raito = len(labels[labels[:] == -1]) / len(labels)
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    
    return Data, n_clusters_, labels

def filter_bare(image):

    """过滤凸点。

        通过遍历每个非零像素点的九宫格来过滤掉邻近点少于4的点，使得输入的边缘图像更平滑。

        Parameters
        ----------
        image : numpy.array
            灰度图像
    
        Returns
        -------
        numpy.array
            过滤凸点后的灰度图

        """

    x,y = np.where(image!=0)
    for i in range(0,len(x)):
        _x = x[i]
        _y = y[i]
        num = 0
        if(image[_x-1][_y-1]!=0):
            num = num+1
        if(image[_x][_y-1]!=0):
            num = num+1
        if(image[_x+1][_y-1]!=0):
            num = num+1
        if(image[_x-1][_y]!=0):
            num = num+1
        if(image[_x+1][_y]!=0):
            num = num+1
        if(image[_x-1][_y+1]!=0):
            num = num+1
        if(image[_x][_y+1]!=0):
            num = num+1
        if(image[_x+1][_y+1]!=0):
            num = num+1
        if(num<4):
            image[_x][_y] = 0
    return image

def colorSpace(image): ## img will be changed

    """识别小腿皮肤区域。

        将图像的RGB空间转换为YCrCb空间，并通过人类皮肤颜色CrCb值范围筛选出小腿皮肤区域。

        Parameters
        ----------
        image : numpy.array
            RGB图像
    
        Returns
        -------
        numpy.array
            筛选后的RGB图像

        """

    b,g,r = cv2.split(image)  
    img = cv2.merge([r,g,b])
    blur = cv2.GaussianBlur(img, (5, 5), 0)

    img_yuv = cv2.cvtColor(blur,cv2.COLOR_RGB2YCrCb)
    index = img_yuv[:,:,1]<133
    img[index] = 0
    index = img_yuv[:,:,1]>173
    img[index] = 0
    index = img_yuv[:,:,2]<77
    img[index] = 0
    index = img_yuv[:,:,2]>127
    img[index] = 0

    return img