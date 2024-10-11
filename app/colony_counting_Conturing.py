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

def count_colonies2(self, Test=False):
        # Prompt user to select the location of cropped pictures
        cropped_images_path = filedialog.askdirectory(title="Select the location of cropped images",
                                                    initialdir=os.path.abspath("CroppedImages"))
        if not cropped_images_path:
            print("No cropped images selected.")
            return
        
        output_directory = filedialog.askdirectory(title="Select the directory to save counting data",
                                               initialdir=os.path.abspath("BlobInfo"))
        if not output_directory:
            print("No output directory selected.")
            return


        # Prompt user to modify blob detection parameters
        params_dialog = tk.Toplevel(self.root)
        params_dialog.title("Modify Blob Detection Parameters")

        # Set default values for blob detection parameters
        min_area_default=300
        max_area_default=4000
        circularity_threshold = 0.5
        bin_width = 100

        min_size_label = tk.Label(params_dialog, text="Min Size:")
        min_size_entry = tk.Entry(params_dialog)
        min_size_entry.insert(0, str(min_area_default))  # Set default value
        min_size_label.grid(row=0, column=0, padx=5, pady=5)
        min_size_entry.grid(row=0, column=1, padx=5, pady=5)

        max_size_label = tk.Label(params_dialog, text="Max Size:")
        max_size_entry = tk.Entry(params_dialog)
        max_size_entry.insert(0, str(max_area_default))  # Set default value
        max_size_label.grid(row=1, column=0, padx=5, pady=5)
        max_size_entry.grid(row=1, column=1, padx=5, pady=5)

        circularity_threshold_label = tk.Label(params_dialog, text="circularity Threshold:")
        circularity_threshold_entry = tk.Entry(params_dialog)
        circularity_threshold_entry.insert(0, str(circularity_threshold))  # Set default value
        circularity_threshold_label.grid(row=2, column=0, padx=5, pady=5)
        circularity_threshold_entry.grid(row=2, column=1, padx=5, pady=5)

        bin_width_label = tk.Label(params_dialog, text="Bin Size:")
        bin_width_entry = tk.Entry(params_dialog)
        bin_width_entry.insert(0, str(bin_width))  # Set default value
        bin_width_label.grid(row=3, column=0, padx=5, pady=5)
        bin_width_entry.grid(row=3, column=1, padx=5, pady=5)

        def count_cells(Test=False):
            # Get user-modified blob detection parameters
            try:
                min_size = int(min_size_entry.get())
                max_size = int(max_size_entry.get())
                circularity_threshold = float(circularity_threshold_entry.get())

            except ValueError:
                print("Invalid input for blob detection parameters. Using default values.")
                params_dialog.destroy()
                return

            params_dialog.destroy()

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

                    # Read the image
                    img = cv2.imread(img_path)

                    # Convert the image to HSV color space
                    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

                    # Define the lower and upper bounds for blue color in HSV
                    lower_blue = np.array([100, 50, 50])
                    upper_blue = np.array([140, 255, 255])


                    # lower_blue = np.array([60, 70, 100])
                    # upper_blue = np.array([130, 255, 255])
                    # Threshold the image to get a binary mask of blue regions
                    mask = cv2.inRange(hsv, lower_blue, upper_blue)

                    # Use MSER to detect regions in the binary mask
                    mser = cv2.MSER_create()
                    regions, _ = mser.detectRegions(mask)

                    # Filter regions based on circularity and minimum area
                    filtered_blobs = []
                    Are_of_blobs = []
                    for region in regions:
                        # Convert region to a polygon
                        hull = cv2.convexHull(region.reshape(-1, 1, 2))

                        # Calculate area and perimeter of the region
                        area = cv2.contourArea(hull)

                        perimeter = cv2.arcLength(hull, True)

                        # Calculate circularity
                        circularity = 4* np.pi * area / (perimeter * perimeter)

                        # Define a minimum area threshold (adjust as needed)
                        if area > min_size and circularity > circularity_threshold and area < max_size:
                            filtered_blobs.append(hull)
                            Are_of_blobs.append(area)

                    # Define the output directory for blob information
                    output_dir = os.path.join(output_directory, 'Counting_Results')
                    os.makedirs(output_dir, exist_ok=True)
                    # Save blob coordinates to CSV file
                    blob_info_path = os.path.join(output_dir, f'BlobInfo_{os.path.basename(img_path)[:-4]}.csv')
                    with open(blob_info_path, 'w', encoding='UTF8', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([len(filtered_blobs), filename])
                        for blob in Are_of_blobs:
                            writer.writerow([blob])
                    blob_image_path = os.path.join(output_dir, f'BlobInfo_{os.path.basename(img_path)[:-4]}.png')
                    # Draw the detected and filtered blobs on the original image
                    cv2.polylines(img, filtered_blobs, 1, (0, 255, 0), 2)
                    print(len(filtered_blobs))
                    # Save the result
                    cv2.imwrite(blob_image_path, img)

                    # Create histogram with specified bin width and range
                    bin_width = 100
                    bins = np.arange(0, 3000 + bin_width, bin_width)

                    # Compute histogram and normalize by area
                    counts, bin_edges = np.histogram(Are_of_blobs, bins=bins)

                    # Compute bin centers
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

                    # Compute uncertainties as square root of counts
                    uncertainties = np.sqrt(counts)

                    plt.bar(bin_centers, counts, width=bin_width, yerr=uncertainties, edgecolor='black', alpha=0.5, label='none')
                    # Labeling and displaying the plot
                    plt.xlabel('Area of Cell Colonies [pixels]')
                    plt.ylabel('Counts')
                    plt.grid()
                    blob_pic_path = os.path.join(output_dir, f'BlobInfo_{os.path.basename(img_path)[:-4]}.jpg')
                    plt.axvline(x=min_size, color='red', linestyle='--', alpha=0.5, label='Cell Counting Cut-off')
                    plt.savefig(blob_pic_path, dpi=1000)
                    plt.clf()
                    # Update progress bar
                    progress["value"] = (idx + 1) / total_images * 100
                    count_window.update_idletasks()
                if Test==True:
                    break
            if Test == True:
                progress.destroy()
                # Display image
                blob_pic_image = Image.open(blob_pic_path)
                blob_pic_image.thumbnail((600, 600))
                tk_blob_pic_image = ImageTk.PhotoImage(blob_pic_image)

                if hasattr(self, "blob_pic_canvas"):
                    self.blob_pic_canvas.destroy()

                self.blob_pic_canvas = tk.Canvas(self.root, width=tk_blob_pic_image.width(), height=tk_blob_pic_image.height())
                self.blob_pic_canvas.pack()
                self.blob_pic_canvas.create_image(0, 0, anchor=tk.NW, image=tk_blob_pic_image)
                self.blob_pic_canvas.image = tk_blob_pic_image

                # Display histogram
                histogram_image = Image.open(blob_Hist_path)
                histogram_image.thumbnail((600, 600))
                tk_histogram_image = ImageTk.PhotoImage(histogram_image)

                if hasattr(self, "histogram_canvas"):
                    self.histogram_canvas.destroy()

                self.histogram_canvas = tk.Canvas(self.root, width=tk_histogram_image.width(), height=tk_histogram_image.height())
                self.histogram_canvas.pack()
                self.histogram_canvas.create_image(0, 0, anchor=tk.NW, image=tk_histogram_image)
                self.histogram_canvas.image = tk_histogram_image

            # Destroy the progress bar once the process is complete
            progress.destroy()
            count_window.destroy()
            print("Cell Counting for all images complete.")

        # Button to apply blob detection parameters
        apply_params_button = tk.Button(params_dialog, text="Apply Parameters", command=count_cells)
        apply_params_button.grid(row=5, column=0, columnspan=2, pady=10)

        # Button to apply blob detection parameters on test
        apply_params_button = tk.Button(params_dialog, text="Test Parameters", command=lambda dialog=params_dialog: count_cells(dialog, Test=True))
        apply_params_button.grid(row=6, column=0, columnspan=2, pady=10)
