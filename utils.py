import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import os
import cv2


def open_idx(df: pd.DataFrame, root_path, i):
    file_name = df['NombreArchivo'][i]
    folder = df['Conjunto'][i]
    return Image.open(os.path.join(root_path, folder, file_name))


def show_bgr_image_in_plt(img):
    return plt.imshow(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
