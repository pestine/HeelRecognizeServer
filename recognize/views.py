from os import times
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from recognize.source.findPoints import findPoints
import json
import cv2
import requests
import time 

### 保证算法的输入图像格式为(224,224,3)
def preprocess(image):
    x,y = image.shape[0:2]
    if(x>y):
        diff = int((x-y)/2)
        image = image[diff:x-diff,:,:]
    else:
        diff = int((y-x)/2)
        image = image[:,diff:y-diff,:]
    
    return cv2.resize(image,(224,224))

def downloadImg(url):
    _t = time.time()
    r = requests.get(url, stream=True, verify=False)
    print('Download code: {}'.format(r.status_code)) # 返回状态码
    if r.status_code == 200:
        image_name = url.split('/')[-1][-10:]
        image_name = 'media/origin/'+image_name+'.jpg'
        open(image_name, 'wb').write(r.content) # 将内容写入图片
        print("Download time: {}".format(time.time()-_t))
    del r
    return image_name

def recognize(url):
    image_name = downloadImg(url)
    image = cv2.imread(image_name)
    image_name = image_name.split('/')[-1]
    image = preprocess(image)
    points = findPoints(image)

    ### 保存图片识别结果
    cv2.circle(image,(points[0][0][1],points[0][0][0]),2,(255,0,0),-1)
    cv2.circle(image,(points[0][0][1],points[0][0][0]),2,(255,0,0),-1)
    cv2.circle(image,(points[0][1][1],points[0][1][0]),2,(255,0,0),-1)
    cv2.circle(image,(points[1][0][1],points[1][0][0]),2,(0,255,0),-1)
    cv2.circle(image,(points[1][1][1],points[1][1][0]),2,(0,255,0),-1)
    cv2.circle(image,(points[2][0][1],points[2][0][0]),2,(0,0,255),-1)
    cv2.circle(image,(points[2][1][1],points[2][1][0]),2,(0,0,255),-1)
    cv2.imwrite("media/image/"+image_name, image)

    dict = {'code':0, 
            'msg':'识别成功',
            'AnkleInfo':{'AnkleResultURL':'media/image/'+image_name,
                         'leftAnkleInfo':{'top':{   'x':str(points[0][0][0]),
                                                    'y':str(points[0][0][1])        
                                          },
                                          'middle':{'x':str(points[1][0][0]),
                                                    'y':str(points[1][0][1])
                                          },
                                          'bottom':{'x':str(points[2][0][0]),
                                                    'y':str(points[2][0][1])
                                          }
                         },
                         'rightAnkleInfo':{'top':{  'x':str(points[0][1][0]),
                                                    'y':str(points[0][1][1])        
                                          },
                                          'middle':{'x':str(points[1][1][0]),
                                                    'y':str(points[1][1][1])
                                          },
                                          'bottom':{'x':str(points[2][1][0]),
                                                    'y':str(points[2][1][1])
                                          }
                         }
            }
    }

    
    return dict

def uploadUrl(request):
    print("postBody: {}".format(request.POST))
    url = request.POST.get('doubleAnkleURL','')
    print("url: {}".format(url))
    respon = json.dumps(recognize(url))

    return HttpResponse(respon)

def uploadImage(request):
    print("postBody: {}".format(request.POST))
    return HttpResponse(request.POST)