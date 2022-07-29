import cv2
from cv2 import waitKey
img = cv2.imread("/home/villads/Documents/Proton_Borh_V79/P4.jpg")
crop_factor = 15
img = cv2.resize(img, (round(img.shape[1]/crop_factor), round(img.shape[0]/crop_factor)))
#img2 = cv2.imread("flower.jpg")

# variables
ix = -1
iy = -1
drawing = False

def draw_reactangle_with_drag(event, x, y, flags, param):
    global ix, iy, drawing, img
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
        RectangleCorr.append(ix)
        RectangleCorr.append(iy)
        RectangleCorr.append(x)
        RectangleCorr.append(y)
        print(RectangleCorr)
    
    

cv2.namedWindow(winname= "Title of Popup Window")
cv2.setMouseCallback("Title of Popup Window", draw_reactangle_with_drag)


while True:
    cv2.imshow("Title of Popup Window", img)
    
    if cv2.waitKey(10) == 27:
        break
    elif waitKey(10) == 13:
        print('Hello :D')
cv2.destroyAllWindows()