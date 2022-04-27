from genericpath import exists
import cv2
import numpy as np
import os
import csv
from skimage.feature import blob_log
from math import sqrt
import matplotlib.pyplot as plt


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
                         threshold=threshold, log_scale=True)
    blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

    if verbose:
        ax = plt.axes()
        plt.imshow(image)
        plot_circles(circle_list=blobs_log, ax=ax)

    return blobs_log


def CropImages(imagepath):
    pattern = ".jpg"
    matching_files = [f for f in os.listdir(imagepath) if pattern in f]
    for pic in matching_files:
        if exists(imagepath + pic[0:len(pic)-4] + "_cropped.jpg") or pic[len(pic)-12:len(pic)] == "_cropped.jpg":
            print(pic + " Already cropped")
            continue
        img = cv2.imread(imagepath + pic)

        crop_img = img[8000:11167, 3850:5842]
        cv2.imwrite(imagepath + pic[0:len(pic)-4] + "_cropped.jpg", crop_img)


def CountCells(datadir, outputdir):
    pattern = "_cropped.jpg"
    matching_files = [f for f in os.listdir(datadir) if pattern in f]
    with open(outputdir + '/Cellcount.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)

        for pic in matching_files:
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
                image_edged = cv2.dilate(image_edged, None, iterations=1)
                image_edged = cv2.erode(image_edged, None, iterations=1)

                # sharpen_kernel = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])
                # sharpen = cv2.filter2D(gray_blurred, -1, sharpen_kernel)

                min_size = 12
                max_size = 60
                threshold = 0.12
                num_sigma = 10
                overlap = 0.3
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


def main():

    # give path to picture folder:
    datapath = "/home/villads/Documents/Cell-Data/"

    for dir in os.listdir(datapath):
        CropImages(datapath + dir + '/')
        # check is output dir exist or create one
        if not os.path.isdir(dir + '_output'):
            os.mkdir(dir + '_output')
        outputdir = dir + '_output'
        CountCells(datapath + dir + '/', outputdir)


if __name__ == "__main__":
    main()
