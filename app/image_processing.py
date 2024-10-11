import cv2
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog
import csv
import os

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
