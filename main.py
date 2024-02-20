import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import csv
import os
from skimage.feature import blob_log
from math import sqrt
import matplotlib.pyplot as plt
import configparser
import tkinter.ttk as ttk

Image.MAX_IMAGE_PIXELS = None

class ColonyCounterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Colony Counter App")

        # Variables
        self.image_path = self.load_last_image_path()
        self.colony_images = []
        self.rectangles = [] 

        # UI Elements
        self.label = tk.Label(root, text="Step 1: Load Image")
        self.label.pack()

        self.load_button = tk.Button(root, text="Load Image", command=self.load_image)
        self.load_button.pack()

        self.crop_button = tk.Button(root, text="Crop Image", command=self.crop_colonies)
        self.crop_button.pack()

        self.count_button = tk.Button(root, text="Count Colonies", command=self.count_colonies)
        self.count_button.pack()

        self.count_button = tk.Button(root, text="Count Colonies 2 method", command=self.count_colonies2)
        self.count_button.pack()

    def load_image(self):
        initial_dir = os.path.dirname(self.image_path) if self.image_path else os.getcwd()
        self.image_path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[("Image files", ".jpg", ".png")],
            initialdir=initial_dir
        )

        if self.image_path:
            self.display_image()
            self.save_last_image_path()

    def display_image(self):
        image = Image.open(self.image_path)
        image.thumbnail((300, 300))
        tk_image = ImageTk.PhotoImage(image)

        # Display image
        if hasattr(self, "canvas"):
            self.canvas.destroy()

        self.canvas = tk.Canvas(self.root, width=tk_image.width(), height=tk_image.height())
        self.canvas.pack()
        self.canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)
        self.canvas.image = tk_image

    def save_last_image_path(self):
        config = configparser.ConfigParser()
        config['LastImage'] = {'Path': self.image_path}
        with open('config.ini', 'w') as configfile:
            config.write(configfile)

    def load_last_image_path(self):
        config = configparser.ConfigParser()
        config.read('config.ini')
        return config.get('LastImage', 'Path', fallback='')

    def crop_colonies(self):
        if not self.image_path:
            print("Please load an image first.")
            return

        # Create a separate window for cropping
        self.crop_window = tk.Toplevel(self.root)
        self.crop_window.title("Crop Colonies")

        # Load the image
        img = cv2.imread(self.image_path)
        self.img_height, self.img_width, _ = img.shape
        self.crop_factor = round(self.img_width/600)

        img = cv2.resize(img, (round(self.img_width/self.crop_factor), round(self.img_height/self.crop_factor)))

        # Convert the image to RGB format
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Convert the image to PhotoImage format
        tk_img = ImageTk.PhotoImage(Image.fromarray(img_rgb))

        # Create a canvas for drawing rectangles
    
        self.crop_canvas = tk.Canvas(self.crop_window, width=round(self.img_width/self.crop_factor), height=round(self.img_height/self.crop_factor))
        self.crop_canvas.pack()

        # Display the image on the canvas
        self.crop_canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)
        self.crop_canvas.image = tk_img

        # Button to apply cropping
        self.apply_crop_button = tk.Button(self.crop_window, text="Apply Crop", command=self.apply_crop)
        self.apply_crop_button.pack()

        # Variables for drawing rectangles
        self.ix, self.iy = -1, -1
        self.drawing = False
        self.rectangle_coords = []

        def draw_rectangle_with_drag(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.drawing = True
                self.ix, self.iy = x, y

            elif event == cv2.EVENT_MOUSEMOVE:
                if self.drawing:

                    # Delete existing rectangles and create a new one
                    self.crop_canvas.delete("rectangle")
                    self.crop_canvas.create_rectangle(self.ix, self.iy, x, y, outline="red", tags="rectangle")

            elif event == cv2.EVENT_LBUTTONUP:
                self.drawing = False
                # self.crop_canvas.delete("rectangle")
                # x, y = min(x, self.img_width), min(y, self.img_height)
                scaled_x = int(x * self.crop_factor)
                scaled_y = int(y * self.crop_factor)
                scaled_ix = int(self.ix * self.crop_factor)
                scaled_iy = int(self.iy * self.crop_factor)
                self.rectangle_coords.append([scaled_ix, scaled_iy, scaled_x, scaled_y])



        cv2.namedWindow("Crop Colonies")
        cv2.setMouseCallback("Crop Colonies", draw_rectangle_with_drag)

        while True:
            cv2.imshow("Crop Colonies", img)
            if cv2.waitKey(10) == 27:
                break

        cv2.destroyAllWindows()

    def apply_crop(self):
        if not self.rectangle_coords:
            print("No crop coordinates found.")
            return
        
        save_directory = filedialog.askdirectory(title="Select the directory to save cropped images and crop information",
                                                 initialdir=os.path.abspath("CroppedImages"))
        if not save_directory:
            print("No save directory selected.")
            return

        # Create directories if they don't exist
        crop_info_dir = 'CropInfo'
        cropped_images_dir = 'CroppedImages'
        
        for directory in [crop_info_dir, cropped_images_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)

        # Ensure rectangle_coords is a list of lists
        if not isinstance(self.rectangle_coords[0], list):
            self.rectangle_coords = [self.rectangle_coords]

        # Apply cropping to the original image for each rectangle
        for idx, coords in enumerate(self.rectangle_coords):
            if len(coords) == 2:
                # Special case: single pair of coordinates
                x, y = coords
                x2, y2 = self.img_width, self.img_height
            else:
                # Normal case: four coordinates
                x, y, x2, y2 = coords
            
            # # Ensure coordinates are within image bounds
            # x, y =max(0, x), min(0, y)
            # x2, y2 = max(self.img_width, x2), max(self.img_height, y2)
                
            # Apply cropping to the original image
            crop_img = cv2.imread(self.image_path)
            crop_img = crop_img[y:y2, x:x2]

            # Define the directory for cropped images
            cropped_images_dir = os.path.join(save_directory, "CroppedImages")
            os.makedirs(cropped_images_dir, exist_ok=True)

            # Save cropped image
            cropped_img_path = os.path.join(cropped_images_dir, f'CroppedImage_{os.path.basename(self.image_path)[:-4]}_{idx + 1}.jpg')
            cv2.imwrite(cropped_img_path, crop_img)

            # Define the directory for crop information
            crop_info_dir = os.path.join(save_directory, "CropInfo")
            os.makedirs(crop_info_dir, exist_ok=True)

            # Save crop information to CSV file
            crop_info_path = os.path.join(crop_info_dir, f'CropInfo_{os.path.basename(self.image_path)[:-4]}_{idx + 1}.csv')
            with open(crop_info_path, 'w', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([x, y, x2, y2])

        # Close crop window
        self.crop_window.destroy()



    def count_colonies(self):
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
        default_min_size = 20
        default_max_size = 30
        default_threshold = 0.11
        default_num_sigma = 10
        default_overlap = 0.3

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
        def count_cells():
            # Get user-modified blob detection parameters
            try:
                min_size = int(min_size_entry.get())
                max_size = int(max_size_entry.get())
                threshold = float(threshold_entry.get())
                num_sigma = int(num_sigma_entry.get())
                overlap = float(overlap_entry.get())
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
                    blobs = self.search_for_blobs(gray_blurred, min_size, max_size, num_sigma, overlap, threshold, verbose=False)

                    # Define the output directory for blob information
                    output_dir = os.path.join(output_directory, 'Counting_Info')
                    os.makedirs(output_dir, exist_ok=True)

                    # Save blob coordinates to CSV file
                    blob_info_path = os.path.join(output_dir, f'BlobInfo_{os.path.basename(img_path)[:-4]}.csv')
                    with open(blob_info_path, 'w', encoding='UTF8', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([len(blobs), filename])
                        for blob in blobs:
                            x, y, r = blob
                            writer.writerow([x, y, r])

                   # Draw circles on the image and save it
                    fig, ax = plt.subplots()
                    ax.imshow(img)
                    plot_circles(circle_list=blobs, ax=ax)
                    blob_pic_path = os.path.join(output_dir, f'BlobInfo_{os.path.basename(img_path)[:-4]}.jpg')
                    plt.savefig(blob_pic_path, dpi=1000)
                    plt.close()

                    print(f"Cell Counting Complete for {filename}")

                    # Update progress bar
                    progress["value"] = (idx + 1) / total_images * 100
                    count_window.update_idletasks()

            # Destroy the progress bar once the process is complete
            progress.destroy()
            count_window.destroy()
            print("Cell Counting for all images complete.")

        # Button to apply blob detection parameters
        apply_params_button = tk.Button(params_dialog, text="Apply Parameters", command=count_cells)
        apply_params_button.grid(row=5, column=0, columnspan=2, pady=10)
    def count_colonies2(self):
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


        def count_cells():
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
                    plt.savefig('combined_histograms.pdf')
                    blob_pic_path = os.path.join(output_dir, f'BlobInfo_{os.path.basename(img_path)[:-4]}.jpg')
                    plt.axvline(x=min_size, color='red', linestyle='--', alpha=0.5, label='Cell Counting Cut-off')
                    plt.savefig(blob_pic_path, dpi=1000)
                    plt.clf()
                    # Update progress bar
                    progress["value"] = (idx + 1) / total_images * 100
                    count_window.update_idletasks()

            # Destroy the progress bar once the process is complete
            progress.destroy()
            count_window.destroy()
            print("Cell Counting for all images complete.")

        # Button to apply blob detection parameters
        apply_params_button = tk.Button(params_dialog, text="Apply Parameters", command=count_cells)
        apply_params_button.grid(row=5, column=0, columnspan=2, pady=10)

    def search_for_blobs(self, image, min_size=3, max_size=15, num_sigma=10, overlap=0.5, threshold=0.02, verbose=True):
        # Detect blobs
        blobs_log = blob_log(image, max_sigma=max_size, min_sigma=min_size, num_sigma=num_sigma, overlap=overlap,
                             threshold=threshold, log_scale=False)
        blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

        return blobs_log

    def search_for_blobs(self, image, min_size=3, max_size=15, num_sigma=10, overlap=0.5, threshold=0.02, verbose=True):
        # Detect blobs
        blobs_log = blob_log(image, max_sigma=max_size, min_sigma=min_size, num_sigma=num_sigma, overlap=overlap,
                             threshold=threshold, log_scale=False)
        blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

        return blobs_log


if __name__ == "__main__":
    root = tk.Tk()
    app = ColonyCounterApp(root)
    root.mainloop()
