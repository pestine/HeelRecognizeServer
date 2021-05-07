import cv2
import copy
import numpy as np
from recognize.source.preprocessing import edgeWithSobel, dbscan, colorSpace

def findContours(image): # image with shape (m,n,3)
    """获取小腿外部轮廓与内部轮廓的信息

        获取外部轮廓和内部轮廓的步骤为：\n
        1.调用preprocessing.colorSpace()识别出图片中腿部皮肤区域。\n
        2.调用preprocessing.colorSpace()将两边小腿区域分离成left和right。\n
        3.对left和right进行腐蚀和膨胀，平滑区域边缘曲线。\n
        4.对left和right分别调用edgeWithSobel()获取边缘信息edgesXL和edgesXR。\n
        5.对left和right分别调用findOuterContours()和findInnerContours()分离小腿外部轮廓特征和小腿跟腱处特征。\n


        Parameters
        ----------
        image : numpy.array
            RGB图像

        Returns
        -------
        list
            contours[left[outer, inner], right[outer, inner], edges[left, right]]

        """

    image = colorSpace(image)

    image_binary = np.zeros_like(image[:,:,0])
    index0 = image[:,:,0]!=0
    index1 = image[:,:,1]!=0
    index2 = image[:,:,2]!=0
    image_binary[index0] = 255
    image_binary[index1] = 255
    image_binary[index2] = 255
    Data, n_clusters_, labels = dbscan(image_binary, 1, 1)

    left_cluster = [] ## left is just first max
    left_i = 0
    for i in range(0,n_clusters_):
        one_cluster = Data[labels==i]
        if(len(one_cluster)>len(left_cluster)):
            left_cluster = one_cluster
            left_i = i
    
    right_cluster = []
    right_i = 0
    for i in range(0,n_clusters_):
        one_cluster = Data[labels==i]
        if(len(one_cluster)>len(right_cluster) and i!=left_i):
            right_cluster = one_cluster
            right_i = i

    left_image_binary = np.zeros_like(image[:,:,0])
    left_image = np.zeros_like(image)
    for point in left_cluster:
        left_image[point[0]][point[1]] = image[point[0]][point[1]]
        left_image_binary[point[0]][point[1]] = 255

    right_image_binary = np.zeros_like(image[:,:,0])
    right_image = np.zeros_like(image)
    for point in right_cluster:
        right_image[point[0]][point[1]] = image[point[0]][point[1]]
        right_image_binary[point[0]][point[1]] = 255

    ##### dilate erode
    kernel = np.ones((3,3),np.uint8)
    left_image = cv2.morphologyEx(left_image, cv2.MORPH_CLOSE, kernel)
    right_image = cv2.morphologyEx(right_image, cv2.MORPH_CLOSE, kernel)

    edgesXL, edgesYL, edgesXYL = edgeWithSobel(left_image)
    edgesXR, edgesYR, edgesXYR = edgeWithSobel(right_image)
    
    ### cut
    left_image[0:int(left_image.shape[0]/4)] = 0 
    right_image[0:int(right_image.shape[0]/4)] = 0
    left_image_binary[0:int(left_image_binary.shape[0]/4)] = 0 
    right_image_binary[0:int(right_image_binary.shape[0]/4)] = 0  
    edgesXL[0:int(edgesXL.shape[0]/5)] = 0 
    edgesXR[0:int(edgesXR.shape[0]/5)] = 0 

    left_o = findOuterContours(left_image_binary)
    right_o = findOuterContours(right_image_binary)
    left_i = findInnerContours(edgesXL, left_image_binary)
    right_i = findInnerContours(edgesXR, right_image_binary)
    contours = [[left_o, left_i], [right_o, right_i], [edgesXL, edgesXR]]

    return contours  # contours[left[outer, inner], right[outer, inner], edges[left, right]]

def findOuterContours(image):  # image is binary with shape (m,n)

    """识别腿部外部轮廓的像素点坐标。

        通过单边腿的二值图找出外轮廓像素点坐标集。

        Parameters
        ----------
        image : numpy.array
            腿部二值图

        Returns
        -------
        list
           腿部外部轮廓的像素点坐标集

        """

    edge_points = [np.where(image[0]!=0)]
   
    for line in image[1:]:
        edge_points.append(np.where(line!=0))
    
    points = [] ## [[x, firsty,lasty]]
    for line in range(len(edge_points)):
        if(len(edge_points[line][0])>0):
            points.append([line, edge_points[line][0][0], edge_points[line][0][-1]])

    return points  ### [] when without point in image


def findInnerContours(edges, image): # binary image with shape (m,n) / edges is the result of sobel with shape(m,n)
    
    """去掉腿部外轮廓，提取足跟处轮廓特征。

        通过单边腿的二值图找出外轮廓像素点坐标集。

        Parameters
        ----------
        edges : numpy.array
            边缘特征图像
        image : numpy.array
            腿部二值图

        Returns
        -------
        numpy.array
           去掉腿部外轮廓的边缘特征图像

        """
    
    edges_f = copy.deepcopy(edges)
    x,y = np.where(edges!=0)
    x_start = x[0]+int((x[-1]-x[0])/16)
    x_end = x[-1]-int((x[-1]-x[0])/16)
    for i in range(x[0],x_start):
        edges_f[i] = 0
    for i in range(x_end,x[-1]):
        edges_f[i] = 0

    edges_f[0:int(image.shape[0]/4)] = 0   # Actually, the second point is below the 1/4 part of picture

    return edges_f
