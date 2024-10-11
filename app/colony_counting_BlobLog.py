from skimage.feature import blob_log
from tkinter import filedialog
from math import sqrt
import tkinter as tk
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tkinter.ttk as ttk
import os
from PIL import Image, ImageTk
import csv
from app.utils import load_last_image_path,save_last_image_path

def count_colonies(self):
    # Prompt user to select the location of cropped pictures
    cropped_images_path = filedialog.askdirectory(title="Select the location of cropped images",
                                                initialdir=load_last_image_path('CroppedImages'))
    save_last_image_path('CroppedImages',cropped_images_path)
    if not cropped_images_path:
        print("No cropped images selected.")
        return
    
    output_directory = filedialog.askdirectory(title="Select the directory to save counting data",
                                            initialdir=load_last_image_path('BlobInfo'))
    save_last_image_path('BlobInfo',output_directory)
    if not output_directory:
        print("No output directory selected.")
        return


    # Prompt user to modify blob detection parameters
    params_dialog = tk.Toplevel(self.root)
    params_dialog.title("Modify Blob Detection Parameters")

    # Set default values for blob detection parameters
    default_min_size = 20
    default_max_size = 30
    default_threshold = 0.11
    default_num_sigma = 10
    default_overlap = 0.3
    bin_width = 100

    min_size_label = tk.Label(params_dialog, text="Min Size:")
    min_size_entry = tk.Entry(params_dialog)
    min_size_entry.insert(0, str(default_min_size))  # Set default value
    min_size_label.grid(row=0, column=0, padx=5, pady=5)
    min_size_entry.grid(row=0, column=1, padx=5, pady=5)

    max_size_label = tk.Label(params_dialog, text="Max Size:")
    max_size_entry = tk.Entry(params_dialog)
    max_size_entry.insert(0, str(default_max_size))  # Set default value
    max_size_label.grid(row=1, column=0, padx=5, pady=5)
    max_size_entry.grid(row=1, column=1, padx=5, pady=5)

    threshold_label = tk.Label(params_dialog, text="Threshold:")
    threshold_entry = tk.Entry(params_dialog)
    threshold_entry.insert(0, str(default_threshold))  # Set default value
    threshold_label.grid(row=2, column=0, padx=5, pady=5)
    threshold_entry.grid(row=2, column=1, padx=5, pady=5)

    num_sigma_label = tk.Label(params_dialog, text="Num Sigma:")
    num_sigma_entry = tk.Entry(params_dialog)
    num_sigma_entry.insert(0, str(default_num_sigma))  # Set default value
    num_sigma_label.grid(row=3, column=0, padx=5, pady=5)
    num_sigma_entry.grid(row=3, column=1, padx=5, pady=5)

    overlap_label = tk.Label(params_dialog, text="Overlap:")
    overlap_entry = tk.Entry(params_dialog)
    overlap_entry.insert(0, str(default_overlap))  # Set default value
    overlap_label.grid(row=4, column=0, padx=5, pady=5)
    overlap_entry.grid(row=4, column=1, padx=5, pady=5)

    bin_width_label = tk.Label(params_dialog, text="Bin Size:")
    bin_width_entry = tk.Entry(params_dialog)
    bin_width_entry.insert(0, str(bin_width))  # Set default value
    bin_width_label.grid(row=5, column=0, padx=5, pady=5)
    bin_width_entry.grid(row=5, column=1, padx=5, pady=5)

    def plot_circles(circle_list, ax, args={"color": "white", "linewidth": 1, "alpha": 0.5}):
        for blob in circle_list:
            y, x, r = blob
            c = plt.Circle((x, y), r, **args, fill=False)
            ax.add_patch(c)
    def FilterBlueColor(img):
        # Convert the BGR color space of image to HSV color space
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Threshold of blue in HSV space
        lower_blue = np.array([60, 70, 100])
        upper_blue = np.array([130, 255, 255])

        # Preparing the mask to overlay
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # Filter only the blue color from the original image using the mask (foreground)
        res = cv2.bitwise_and(img, img, mask=mask)
        return res
    def count_cells(dialog, Save=False):

        # Get user-modified blob detection parameters
        try:
            min_size = int(min_size_entry.get())
            max_size = int(max_size_entry.get())
            threshold = float(threshold_entry.get())
            num_sigma = int(num_sigma_entry.get())
            overlap = float(overlap_entry.get())
            bin_width = int(bin_width_entry.get())
        except ValueError:
            print("Invalid input for blob detection parameters. Using default values.")
            params_dialog.destroy()
            return

        # params_dialog.destroy()

        # Create a separate window for cell counting
        count_window = tk.Toplevel(self.root)
        count_window.title("Count Cells")

        # Create a progress bar
        progress = ttk.Progressbar(count_window, orient="horizontal", length=300, mode="determinate")
        progress.pack()

        # Get the total number of images for calculating progress
        total_images = len([filename for filename in os.listdir(cropped_images_path) if filename.endswith(".jpg")])

        # Iterate through all the cropped images in the selected location
        for idx, filename in enumerate(os.listdir(cropped_images_path)):
            if filename.endswith(".jpg"):
                img_path = os.path.join(cropped_images_path, filename)
                # Load the cropped image
                img = cv2.imread(img_path)
                # Blue Filter
                img_b = FilterBlueColor(img)
                out = cv2.addWeighted(img_b, 2, img_b, 0, 100)

                # Convert image to grayscale
                gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)

                # Apply Gaussian Blur
                gray_blurred = cv2.GaussianBlur(gray, (15, 15), 0)

                # Detect blobs
                blobs_log = blob_log(gray_blurred, max_sigma=max_size, min_sigma=min_size, num_sigma=num_sigma, overlap=overlap,
                            threshold=threshold, log_scale=False)
                blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)
                blobs=blobs_log
                # Define the output directory for blob information
                output_dir = os.path.join(output_directory, 'Counting_Info')
                os.makedirs(output_dir, exist_ok=True)

                # Save blob coordinates to CSV file
                blob_info_path = os.path.join(output_dir, f'BlobInfo_{os.path.basename(img_path)[:-4]}.csv')
                Are_of_blobs = []
                with open(blob_info_path, 'w', encoding='UTF8', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([len(blobs), filename])
                    for blob in blobs:
                        x, y, r = blob
                        area= np.pi * r**2
                        Are_of_blobs.append(area)
                        writer.writerow([x, y, r])

                # Create histogram with specified bin width and range
                bins = np.arange(0, 3000 + bin_width, bin_width)

                # Compute histogram and normalize by area
                counts, bin_edges = np.histogram(Are_of_blobs)

                # Compute bin centers
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

                # Compute uncertainties as square root of counts
                uncertainties = np.sqrt(counts)

                plt.bar(bin_centers, counts, width=bin_width, yerr=uncertainties, edgecolor='black', alpha=0.5, label='none')
                # Labeling and displaying the plot
                plt.xlabel('Area of Cell Colonies [pixels]')
                plt.ylabel('Counts')
                plt.grid()
                blob_Hist_path = os.path.join(output_dir, f'BlobHist_{os.path.basename(img_path)[:-4]}.jpg')
                plt.axvline(x=min_size, color='red', linestyle='--', alpha=0.5, label='Cell Counting Cut-off')
                if Save==True:
                    plt.savefig(blob_Hist_path, dpi=1000)
                plt.clf()

                # Draw circles on the image and save it
                fig, ax = plt.subplots()
                ax.imshow(img)
                plot_circles(circle_list=blobs, ax=ax)
                blob_pic_path = os.path.join(output_dir, f'BlobInfo_{os.path.basename(img_path)[:-4]}.jpg')
                if Save==True:
                    plt.savefig(blob_Hist_path, dpi=1000)
                plt.close()

                print(f"Cell Counting Complete for {filename}")

                # Update progress bar
                progress["value"] = (idx + 1) / total_images * 100
                count_window.update_idletasks()
                if Save==True:
                    break
                




        progress.destroy()
        count_window.destroy()
        print("Cell Counting for all images complete.")

    # Button to apply blob detection parameters
    apply_params_button = tk.Button(params_dialog, text="Run", command=lambda dialog=params_dialog: count_cells(dialog, Save=False))
    apply_params_button.grid(row=6, column=0, columnspan=2, pady=10)

    # Button to apply blob detection parameters on test
    apply_params_button = tk.Button(params_dialog, text="Save", command=lambda dialog=params_dialog: count_cells(dialog, Save=True))
    apply_params_button.grid(row=7, column=0, columnspan=2, pady=10)

    # Button to apply blob detection parameters on test
    apply_params_button = tk.Button(params_dialog, text="Exit", command=exit)
    apply_params_button.grid(row=8, column=0, columnspan=2, pady=10)
