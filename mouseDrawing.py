import cv2
from cv2 import waitKey
import csv
import os

def draw_reactangle_with_drag(event, x, y, flags, param):
    global ix, iy, drawing, img, writer
    RectangleCorr = []

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix = x
        iy = y
        
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.rectangle(img, pt1=(ix,iy), pt2=(x, y),color=(0,255,255),thickness=-1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(img, pt1=(ix,iy), pt2=(x, y),color=(0,255,255),thickness=-1)
        if(ix>x and iy>y):
            RectangleCorr.append(y)
            RectangleCorr.append(iy)
            RectangleCorr.append(x)
            RectangleCorr.append(ix)
            crop_img = img[y:iy, x:ix]
        elif(ix>x and iy < y): 
            RectangleCorr.append(iy)
            RectangleCorr.append(y)
            RectangleCorr.append(x)
            RectangleCorr.append(ix)
            crop_img = img[iy:y, x:ix]
        elif(ix<x and iy < y): 
            RectangleCorr.append(iy)
            RectangleCorr.append(y)
            RectangleCorr.append(ix)
            RectangleCorr.append(x)
            crop_img = img[iy:y, ix:x]
        elif(ix<x and iy > y): 
            RectangleCorr.append(y)
            RectangleCorr.append(iy)
            RectangleCorr.append(ix)
            RectangleCorr.append(x)
            crop_img = img[y:iy, ix:x]
        
        writer.writerow(RectangleCorr)
        print(RectangleCorr)


imagepath = "/home/villads/Documents/Cell-Data3/Proton_Borh_V79/"
# variables
ix = -1
iy = -1
drawing = False

pattern = ".jpg"
matching_files = [f for f in os.listdir(imagepath) if pattern in f]
if not os.path.isdir('CropInfo'):
            os.mkdir('CropInfo')
for pic in matching_files:
    img = cv2.imread(imagepath + pic)
    crop_factor = 20
    img = cv2.resize(img, (round(img.shape[1]/crop_factor), round(img.shape[0]/crop_factor)))
    cv2.namedWindow(winname= "Title of Popup Window")
    cv2.resizeWindow("Title of Popup Window", 300, 700) 
    with open('CropInfo/CropInfo_' + pic[0:len(pic)-4] + '.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        cv2.setMouseCallback("Title of Popup Window", draw_reactangle_with_drag)
        while True:
            cv2.imshow("Title of Popup Window", img)      
            if cv2.waitKey(10) == 27:
                break
        cv2.destroyAllWindows()
        f.close()  

 