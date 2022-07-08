from genericpath import exists
import cv2
import numpy as np
import os
import csv
from skimage.feature import blob_log
from math import dist, sqrt
import matplotlib.pyplot as plt
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.draw import circle_perimeter


def main():

    # give path to picture folder:
    datapath = "/home/villads/Documents/Cell-Data2/"


    for dir in os.listdir(datapath):
        if dir[-7:len(dir)] == "_output":
            continue
        CropImages(datapath + dir + '/')
        # check is output dir exist or create one
        if not os.path.isdir(datapath + dir + '_output'):
            os.mkdir(datapath + dir + '_output')
        outputdir = datapath + dir + '_output'
        CountCells(datapath + dir + '/', outputdir)


def plot_circles(circle_list, ax, args={"color": "white", "linewidth": 1, "alpha": 0.5}):
    # print(len(circle_list))
    k = 0
    for blob in circle_list:
        y, x, r = blob
        c = plt.Circle((x, y), r, **args, fill=False)
        ax.add_patch(c)
        k += 1
    # print("This is k {}".format(k))

def search_for_blobs(image, min_size=3, max_size=15, num_sigma=10, overlap=0.5, threshold=0.02, verbose=True):

    # detect blobs
    blobs_log = blob_log(image, max_sigma=max_size, min_sigma=min_size, num_sigma=num_sigma, overlap=overlap,
                         threshold=threshold, log_scale=False)
    blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)
    if verbose:
        ax = plt.axes()
        plt.imshow(image)
        plot_circles(circle_list=blobs_log, ax=ax)

    return blobs_log

def CropImages(imagepath, ContainerType="petri dish"):
    pattern = ".jpg"
    matching_files = [f for f in os.listdir(imagepath) if pattern in f]
    for pic in matching_files:
        if exists(imagepath + pic[0:len(pic)-4] + "_cropped.jpg") or pic[len(pic)-12:len(pic)] == "_cropped.jpg":
            print(pic + " Already cropped")
            continue
        if exists(imagepath + "refference.jpg") or pic[len(pic)-12:len(pic)] == "_cropped.jpg":
            print(pic + " Is a refference picture")
            continue
        print('cropping ', imagepath + pic)
        img = cv2.imread(imagepath + pic)
        crop_factor = 8
        img_small = cv2.resize(img, (round(img.shape[1]/crop_factor), round(img.shape[0]/crop_factor)))
        # cv2.imshow('test',img_small)
        # cv2.waitKey(0)

        petri_D = 1300
        #find cell container
        if ContainerType=="petri dish":
            circles = detect_circle_by_canny(img_small, radius=round(petri_D/crop_factor))
            cropname = 0
            for i in circles:
                x=i[0]*crop_factor 
                y=i[1]*crop_factor
                r=i[2]*crop_factor
                mask = np.zeros(img.shape[:2], dtype="uint8")
                mask = cv2.circle(mask,(x,y),r,255,-1)
                #cv2.imwrite(argv[2],mask)
                out = cv2.bitwise_and(img, img, mask=mask)
                crop_img = out[y-r:y+r, x-r:x+r]
            
                cv2.putText(img=img_small, text=str(cropname), org=(round(x/crop_factor), round(y/crop_factor)), 
                           fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=3, color=(0, 0, 0),thickness=3)
                cv2.imwrite(imagepath  + str(cropname) + "_cropped.jpg", crop_img)
                cropname +=1
            cv2.imwrite(imagepath  + "refference.jpg", img_small)
        else:         
            crop_img = img[8000:11167, 3850:5842]
            cv2.imwrite(imagepath + pic[0:len(pic)-4] + "_cropped.jpg", crop_img)


def CountCells(datadir, outputdir):
    pattern = "_cropped.jpg"
    matching_files = [f for f in os.listdir(datadir) if pattern in f]
    with open(outputdir + '/Cellcount.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)

        for pic in matching_files:
            if(pic[0] != "0"):
                continue
            with open(outputdir + '/' + pic[0:len(pic)-4] + "_blob" + ".csv", 'w', encoding='UTF8', newline='') as fr:
                writer_blob = csv.writer(fr)

                print(datadir + pic)
                # Read image.
                img = cv2.imread(datadir + pic, cv2.IMREAD_COLOR)

                # BlueFilter
                img_b = FilterBlueColor(img)

                # alpha 1  beta 0      --> no change
                # 0 < alpha < 1        --> lower contrast
                # alpha > 1            --> higher contrast
                # -127 < beta < +127   --> good range for brightness values

                # call addWeighted function. use beta = 0 to effectively only operate one one image

                # out = cv2.addWeighted(img_b, 2, img_b, 0, 100)
                # cv2.imshow(pic, cv2.resize(out, (480, 800)))

                # cv2.waitKey(0)
                # Convert to grayscale.
                gray = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

                # Blur using 3 * 3 kernel.
                # gray_blurred = cv2.blur(gray, (5, 5))  # (10, 10)
                gray_blurred = cv2.GaussianBlur(gray, (15, 15), 0)

                # perform edge detection, then perform a dilation + erosion to close gaps in between object edges
                image_edged = cv2.Canny(gray_blurred, 30, 55)
                image_edged = cv2.imfill
                image_edged = cv2.erode(image_edged, None, iterations=1) # multiple times
                blob detection
                image_edged = cv2.dilate(image_edged, None, iterations=1) # active contour

                # sharpen_kernel = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])
                # sharpen = cv2.filter2D(gray_blurred, -1, sharpen_kernel)

                min_size = 12
                max_size = 30
                threshold = 0.11
                num_sigma = 40
                overlap = 0.7
                cv2.imwrite(outputdir + '/' + pic[0:len(pic)-4] + "input.png",image_edged)

                # verbose=True
                blobs = search_for_blobs(image=gray_blurred, min_size=min_size, max_size=max_size, num_sigma=num_sigma,
                                         overlap=overlap, threshold=threshold, verbose=True)
                # print(detected_blobs)
                writer.writerow([len(blobs), pic])
                plt.imshow(img)
                plt.savefig(outputdir + '/' + pic[0:len(pic)-4] + ".png", dpi=1000)
                # plt.show()
                plt.close()
                # print(outputdir + '/' + pic[0:len(pic)-4] + ".png")
                for blob in blobs:
                    x, y, r = blob
                    writer_blob.writerow([x, y, r])
                fr.close()
                print("{} completed".format(pic))

        f.close()

def FilterBlueColor(img):

    # It converts the BGR color space of image to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Threshold of blue in HSV space
    lower_blue = np.array([60, 70, 100])
    upper_blue = np.array([277, 275, 275])

    # preparing the mask to overlay
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Filter only the red colour from the original image using the mask(foreground)
    res = cv2.bitwise_and(img, img, mask=mask)

    return res

def detect_circle_by_canny(image_bw, radius=395):
    img = cv2.medianBlur(image_bw,1)
    cimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(cimg,cv2.HOUGH_GRADIENT,1,20,
                            param1=50,param2=30,minRadius=round(radius-(radius*0.1)),maxRadius=radius)
    circles = np.uint16(np.around(circles))
    cv2.namedWindow('detected circles',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('detected circles', 300, 700)
    cv2.imshow('detected circles',cimg)
    # Remove overlapping circles
    CircleList=[]
    for i in circles[0,:]:
        CircleList.append(i)
    for i in range(len(CircleList)):
        pt1=CircleList[i]
        if (isinstance(pt1, int)):
                continue
        r=pt1[2]
        for k in range(len(CircleList)):
            pt2=CircleList[k]
            if (isinstance(pt2, int) or isinstance(pt1, int)):
                continue
            if(pt1[0]==pt2[0] and pt1[1]==pt2[1]):
                continue
            distVec= [int(pt2[0])-int(pt1[0]),int(pt2[1])-int(pt1[1])]
            distLength=sqrt(pow(distVec[0],2) + pow(distVec[1],2))

            if(distLength < r and distLength!=0):
                if(int(pt1[0]) < int(pt2[0])):
                    CircleList[k]=int(0)
                else:
                    CircleList[i]=int(0)
                
    CircleList = [i for i in CircleList if not isinstance(i, int)]

    # print(CircleList)
    for i in range(len(CircleList)):
        circ=CircleList[i]
        # draw the outer circle
        cv2.circle(cimg,(circ[0],circ[1]),circ[2],(0,255,0),2)
        # draw the center of the circle
        cv2.circle(cimg,(circ[0],circ[1]),2,(0,0,255),3)
    # cv2.imshow('detected circles',cimg)
    # cv2.waitKey(0)

    return CircleList


if __name__ == "__main__":
    main()
