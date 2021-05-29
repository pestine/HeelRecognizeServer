# %%
from recognize.source.contours import findContours
from recognize.source.points import firstPoint, secondPoint, thirdPoint
import numpy as np
from math import acos, pi, sqrt
# %%
def findPoints(image):

    """提取三个特征点

        1.通过调用contour.py中的findContours()来识别轮廓特征。\n
        2.通过调用point.py中的firstPoint(), secondPoint(), thirdPoint()来从轮廓特征中识别三个关键点。

        Parameters
        ----------
        image : numpy.array
            RGB图像

        Returns
        -------
        list
            shape:(3,2,2)\n
           [[left point, right point],\n
           [left point, right point],\n
           [left point, right point]]

        """

    try:
        contours = findContours(image)
    except:
        return [[[0,0],[0,0]],
                [[0,0],[0,0]],
                [[0,0],[0,0]]]
    points = []

    try:
        xL,yL = firstPoint(contours[0][0])
    except:
        xL,yL = 0, 0
    try:
        xR,yR = firstPoint(contours[1][0])
    except:
        xR,yR = 0, 0
    points.append([[xL,yL],[xR,yR]])

    try:
        xL,yL = secondPoint(contours[0][1])
    except:
        xL,yL = 0, 0
    try:
        xR,yR = secondPoint(contours[1][1])
    except:
        xR,yR = 0, 0
    points.append([[xL,yL],[xR,yR]])
    
    try:
        xL,yL = thirdPoint(contours[0][0])
    except:
        xL,yL = 0, 0
    try:
        xR,yR = thirdPoint(contours[1][0])
    except:
        xR,yR = 0, 0
    points.append([[xL,yL],[xR,yR]])

    return points #(3,2,2)


# %%
def angleCalculate(points):

    """计算三个关键点连成的两条线段组成的角度

        1.将关键点坐标组成两个向量，利用向量相关公式计算角度。

        Parameters
        ----------
        points : numpy.array
            关键点的像素点坐标

        Returns
        -------
        int

        """

    points = np.asarray(points)
    angle = [0, 0]

    vl1 = points[1][0]-points[0][0]
    print("vl1:{}".format(vl1))
    vl2 = points[2][0]-points[1][0]
    print("vl2:{}".format(vl2))
    vr1 = points[1][1]-points[0][1]
    print("vr1:{}".format(vr1))
    vr2 = points[2][1]-points[1][1]
    print("vr2:{}".format(vr2))
    
    vl11 = vl1[0]*vl1[0] + vl1[1]*vl1[1]
    vl22 = vl2[0]*vl2[0] + vl2[1]*vl2[1]
    vl12 = vl1[0]*vl2[0] + vl1[1]*vl2[1]
    try:
        angle[0] = 180 - int(acos(vl12/sqrt(vl11)/sqrt(vl22))/pi*180)
    except:
        print("Zero divisor!")

    vr11 = vr1[0]*vr1[0] + vr1[1]*vr1[1]
    vr22 = vr2[0]*vr2[0] + vr2[1]*vr2[1]
    vr12 = vr1[0]*vr2[0] + vr1[1]*vr2[1]
    try:
        angle[1] = 180 - int(acos(vr12/sqrt(vr11)/sqrt(vr22))/pi*180)
    except:
        print("Zero divisor!")
        
    return angle

# %%
def predict(points):

    """预测是否存在足内旋或者足外旋

        1.通过调用angleCalculate()来计算角度。\n

        Parameters
        ----------
        points : numpy.array
            关键点的像素点坐标

        Returns
        -------
        int
        str

        """

    angle = angleCalculate(points)
    prediction = ["Normal", "Normal"]
    if angle[0]>0 and angle[0] < 170:
        prediction[0] = "Pronation"
    if angle[0] > 190:
        prediction[0] = "Supination"
    if angle[1]>0 and angle[1] < 170:
        prediction[1] = "Pronation"
    if angle[1] > 190:
        prediction[1] = "Supination"

    return angle, prediction


