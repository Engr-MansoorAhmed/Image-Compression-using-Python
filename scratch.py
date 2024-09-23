from ctypes import pythonapi
import numpy as np
import pandas as pd
import cv2
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as image
import glob
import tkinter as tk
from sklearn.cluster import MiniBatchKMeans
import os


path = "pexels-pixabay-60597.jpg"
img = cv2.imread(path)
file_info = (os.stat(path).st_size)/1024

img = img/255
img = cv2.resize(img, (300,300))

img1  = img.reshape(300*300,3)

size_param=[]
for i in range(2, 7):
    kmeans = MiniBatchKMeans(i)
    kmeans.fit(img1)
    new_colors = kmeans.cluster_centers_[kmeans.predict(img1)]
    new_img = new_colors.reshape(img.shape)
    # cv2.imwrite(str(i)+"jpg", new_img)
    cv2.imshow(str(i),new_img)
    plt.title("no. of clusters : " + str(i))
    print("no. of clusters : " ,i)



    cv2.waitKey(2000)
    cv2.destroyAllWindows()








