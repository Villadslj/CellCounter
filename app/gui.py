import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from app.utils import load_last_image_path,save_last_image_path
from app.image_processing import crop_colonies
from app.colony_counting_BlobLog import count_colonies
from app.colony_counting_Conturing import count_colonies2
import os

class ColonyCounterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Colony Counter App")
        self.count_colonies = count_colonies.__get__(self)
        self.count_colonies2 = count_colonies2.__get__(self)

        # Variables
        self.image_path = load_last_image_path('LastImage')
        self.colony_images = []
        self.rectangles = [] 

        # UI Elements
        self.label = tk.Label(root, text="Step 1: Load Image")
        self.label.pack()

        self.load_button = tk.Button(root, text="Load Image", command=self.load_image)
        self.load_button.pack()

        self.crop_button = tk.Button(root, text="Crop Image", command=crop_colonies)
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
            save_last_image_path('LastImage',self.image_path + "resources/")


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
