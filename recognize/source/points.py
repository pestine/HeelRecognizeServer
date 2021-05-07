import cv2
import numpy as np
from recognize.source.preprocessing import filter_bare,dbscan

def firstPoint(contour): # contour with shape (n,3) [x,first,last]

    """识别第一个关键点。

        第一个关键点取腿部横向宽度最宽的截线的中点，也就是外轮廓右边缘与左边缘差值的最大值所在截线的中点。

        Parameters
        ----------
        contour : list
            外轮廓坐标点集

        Returns
        -------
        list
            [x, y]\n
            x (int): 第一个关键点的x轴坐标\n
            y (int): 第一个关键点的y轴坐标\n

        """

    #### filter outliers by median
    contour = np.asarray(contour, dtype=float)
    first_md = np.median(contour[: int(contour.shape[0]/2),1])
    last_md = np.median(contour[: int(contour.shape[0]/2),2])

    y = []
    for line in contour[: int(contour.shape[0]/2)]:  # Actually, the first point is above the center of picture
        if(abs(line[1]-first_md)>contour.shape[0]/8 or abs(line[2]-last_md)>contour.shape[0]/8):
            continue
        y.append([line[0], line[1], line[2], line[2]-line[1]])
    y = np.asarray(y, dtype=float)
    max_candidate_list = np.asarray(np.where(y[:, 3]==np.max(y[:,3])))[0]
    x = int(y[max_candidate_list[-1]][0])
    y = int((y[max_candidate_list[-1]][1]+y[max_candidate_list[-1]][2])/2)
    
    return x, y 

def secondPoint(edges): # edges with shape (m,n)

    """识别第二个关键点。

        第二个关键点为跟腱纵向中点，也就是跟腱最窄处的点。识别第二个关键点的步骤为：\n
        1.调用dbscan聚类算法，将跟腱处左右两根边缘分离成left和right。\n
        2.对分离后的left和right进行多次的腐蚀和膨胀，使得边缘曲线更平滑。\n
        3.若分离后的left和right长度差距不大，则取两条边缘距离最近处的点为第二个关键点。\n
        4.若分离后的left和right长度差距过大，则证明两条边缘未能分离，则取长度更大的那条边缘的最窄处为第二个关键点。\n

        Parameters
        ----------
        edges : numpy.array
            跟腱区域的边缘检测的特征图

        Returns
        -------
        list
            [x, y]\n
            x (int): 第二个关键点的x轴坐标\n
            y (int): 第二个关键点的y轴坐标\n

        """

    index = edges < 50
    edges[index] = 0

    kernel3 = np.ones((3,3),np.uint8)
    for i in range(0,1):
        edges = cv2.dilate(edges, kernel3)

    ### dbscan
    Data, n_clusters_, labels = dbscan(edges, 1.5, 1)
    if(n_clusters_ == 0):
        print("No heel detected!")
        return 0,0

    ### find the cluster with max number [n,2]
    left_image = np.zeros((edges.shape[0],edges.shape[1]))
    left_i = -1
    left_cluster = []
    for i in range(n_clusters_):
        one_cluster = Data[labels == i]
        if(len(one_cluster)>len(left_cluster)):
            left_cluster = one_cluster
            left_i = i

    ### find the cluster with max number [n,2]
    right_image = np.zeros((edges.shape[0],edges.shape[1]))
    right_i = -1
    right_cluster = []
    for i in range(n_clusters_):
        if(i==left_i):
            continue
        one_cluster = Data[labels == i]
        if(len(one_cluster)>len(right_cluster)):
            right_cluster = one_cluster
            right_i = i
    
    ###展示最终选择的内轮廓，然后cluster过期了
    for point in left_cluster:
        left_image[point[0]][point[1]] = 255
    for point in right_cluster:
        right_image[point[0]][point[1]] = 255

    ### dilate erode
    kernel3 = np.ones((3,3),np.uint8)
    left_image = cv2.morphologyEx(left_image, cv2.MORPH_CLOSE, kernel3)
    right_image = cv2.morphologyEx(right_image, cv2.MORPH_CLOSE, kernel3)
    
    ### 去掉凸点 
    left_image = filter_bare(left_image)
    right_image = filter_bare(right_image)
    
    ### 新的cluster上线 [2,n]
    left_cluster = np.asarray(np.where(left_image!=0))  
    right_cluster = np.asarray(np.where(right_image!=0))

    ###判断边的左右
    # print("averageL:{}  averageR:{}".format(np.average(left_cluster[1]), np.average(right_cluster[1])))
    if(np.average(left_cluster[1]) > np.average(right_cluster[1])):
        # print("left shift!")
        temp = left_image
        left_image = right_image
        right_image = temp
        temp = left_cluster
        left_cluster = right_cluster
        right_cluster = temp
    
    x = 0
    y = 0

    ###判断识别出混合边缘还是双边缘
    lengthL = 0
    lengthR = 0
    if(len(left_cluster[0])==0 and len(right_cluster[0])==0):
        lengthL = 0
        lengthR = 0
    elif(len(left_cluster[0])==0):
        lengthL = 0
        lengthR = right_cluster[0][-1]-right_cluster[0][0]
    elif(len(right_cluster[0])==0):
        lengthR = 0
        lengthL = left_cluster[0][-1]-left_cluster[0][0]
    else: 
        lengthL = left_cluster[0][-1]-left_cluster[0][0]
        lengthR = right_cluster[0][-1]-right_cluster[0][0]
        
    if(lengthL<=int(lengthR/5) or lengthR<=int(lengthL/5)): #混合边缘
        temp_image = []
        temp_cluster = []
        differ = []  # [[x, left, right, right-left],]
        if(lengthL>lengthR):
            temp_image = left_image
            temp_cluster = left_cluster
        else:
            temp_image = right_image
            temp_cluster = right_cluster

        ###不考虑单边缘的情况
        if(len(temp_cluster[0])==0):
            print("No heel detected!")
            return 0,0

        for i in range(temp_cluster[0][0]+int((temp_cluster[0][-1]-temp_cluster[0][0])/8),\
            temp_cluster[0][-1]-int((temp_cluster[0][-1]-temp_cluster[0][0])/12)): #heel样本的第一层到最后一层
            _x = i
            plist = np.where(temp_image[i]!=0)[0]
            if(len(plist)==0):
                continue
            _left = plist[0]
            _right = plist[-1]
            differ.append([_x, _left, _right, _right-_left])
        differ = np.asarray(differ) 

        if(len(differ)==0 or len(differ[0])==0):
            return 0, 0
        _line = np.where(differ[:,3]==np.min(differ[:,3]))
        _line = int(np.median(_line[0]))
        x = int(differ[_line][0])
        y = int((differ[_line][1]+differ[_line][2])/2)
        for i in range(0,15):
            if(_line+i >= len(differ)):
                break
            if(abs(differ[_line+i][3]-differ[_line][3])<3):
                x = int(differ[_line+i][0])
                y = int((differ[_line+i][1]+differ[_line+i][2])/2)
    else: #双边缘

        ###再次erode cluster再次过期
        kernel3 = np.ones((3,3),np.uint8)
        right_image = cv2.morphologyEx(right_image, cv2.MORPH_CLOSE, kernel3)
        if(len(left_cluster[0])>=1200):
            left_image = cv2.erode(left_image, kernel3)
        if(len(right_cluster[0])>=1200):
            right_image = cv2.erode(right_image, kernel3) 
        ### 新的cluster上线 [2,n]
        left_cluster = np.asarray(np.where(left_image!=0))  
        right_cluster = np.asarray(np.where(right_image!=0))

        differ = []  #[x, left, right, differ]

        ###处理边缘的头尾
        x_start = left_cluster[0][0]
        x_end = left_cluster[0][-1]
        x_start_min = left_cluster[0][0]
        x_end_max = left_cluster[0][-1]
        if(left_cluster[0][0] < right_cluster[0][0]):
            x_start = right_cluster[0][0]   #取第一个x的最大值
        else:
            x_start_min = right_cluster[0][0]
        if(left_cluster[0][-1] > right_cluster[0][-1]):
            x_end = right_cluster[0][-1]    #取最后一个x的最小值
        else:
            x_end_max = right_cluster[0][-1]
        x_start_min = x_start_min+int((x_end_max-x_start_min)/16)
        x_end_max = x_end_max-int((x_end_max-x_start_min)/16)
        if(x_start<x_start_min):
            x_start = x_start_min
        if(x_end>x_end_max):
            x_end = x_end_max
        
        for i in range(x_start,x_end): #heel样本的第一层到最后一层
            _x = i
            listL = np.where(left_image[i]!=0)[0]
            listR = np.where(right_image[i]!=0)[0]
            if(len(listL)==0 and len(listR)==0):
                differ.append([_x, differ[-1][1], differ[-1][2], differ[-1][3]])
            elif(len(listL)==0):
                differ.append([_x, differ[-1][1], listR[0], listR[0]-differ[-1][1]])
            elif(len(listR)==0):
                differ.append([_x, listL[-1], differ[-1][2], differ[-1][2]-listL[-1]])
            else:
                _left = listL[-1]
                _right = listR[0]
                differ.append([_x, _left, _right, _right-_left])
        differ = np.asarray(differ) 
        if(len(differ)==0 or len(differ[0])==0):
            return 0, 0
        _line = np.where(differ[:,3]==np.min(differ[:,3]))
        _line = int(np.median(_line[0]))
        x = int(differ[_line][0])
        y = int((differ[_line][1]+differ[_line][2])/2)

    return x,y

def thirdPoint(contour): # contour with shape (x, first, last)

    """识别第三个关键点。

        第三个关键点取腿部外轮廓在图片中的最低点。

        Parameters
        ----------
        contour : list
            外轮廓坐标点集

        Returns
        -------
        list
            [x, y]\n
            x (int): 第三个关键点的x轴坐标\n
            y (int): 第三个关键点的y轴坐标\n

        """

    x = int(contour[-1][0])
    y = int((contour[-1][2]+contour[-1][1])/2)
    return x, y
