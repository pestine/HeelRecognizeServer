# %%
from recognize.source.contours import findContours
from recognize.source.points import firstPoint, secondPoint, thirdPoint

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



