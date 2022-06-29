import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import pandas as pd
from colonycounter import Counter
import PIL


path = "/home/villads/Documents/Mixed beam exp day 2 - Scan 1.jpg"
PIL.Image.MAX_IMAGE_PIXELS = 933120000
counter = Counter(image_path=path)
counter.detect_area_by_canny()

