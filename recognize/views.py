from os import times
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from recognize.source.findPoints import findPoints
import json
import cv2
import requests
import time
import os

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

def resultHandling(image, image_name, points):
    ### save result as image
    print(points)
    cv2.circle(image,(points[0][0][1],points[0][0][0]),2,(255,0,0),-1)
    cv2.circle(image,(points[0][0][1],points[0][0][0]),2,(255,0,0),-1)
    cv2.circle(image,(points[0][1][1],points[0][1][0]),2,(255,0,0),-1)
    cv2.circle(image,(points[1][0][1],points[1][0][0]),2,(0,255,0),-1)
    cv2.circle(image,(points[1][1][1],points[1][1][0]),2,(0,255,0),-1)
    cv2.circle(image,(points[2][0][1],points[2][0][0]),2,(0,0,255),-1)
    cv2.circle(image,(points[2][1][1],points[2][1][0]),2,(0,0,255),-1)
    cv2.imwrite("media/image/"+image_name, image)

    dict = {'code':200, 
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

def recognizeUrl(url):
    image_path = downloadImg(url)
    image = cv2.imread(image_path)
    image_name = image_path.split('/')[-1]
    image = preprocess(image)
    points = findPoints(image)
    return resultHandling(image, image_name, points)

def recognizeImage(image_path):
    image = cv2.imread(image_path)
    image_name = image_path.split('/')[-1]
    image = preprocess(image)
    points = findPoints(image)
    return resultHandling(image, image_name, points)

def uploadUrl(request):
    print("postBody: {}".format(request.POST))
    url = request.POST.get('doubleAnkleURL','')
    print("url: {}".format(url))
    respon = json.dumps(recognizeUrl(url))

    return HttpResponse(respon)

def uploadImage(request):
    print("postBody: {}".format(request))
    file_obj = request.FILES.get("image")

    print("file_obj", file_obj.name)
    file_path = 'media/origin/' + file_obj.name
    print("file_path", file_path)
 
    with open(file_path, 'wb+') as f:
      for chunk in file_obj.chunks():
        f.write(chunk)
    
    respon = json.dumps(recognizeImage(file_path))
    
    return HttpResponse(respon)