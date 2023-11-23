import os
from os.path import join
import cv2 as cv
import numpy as np
from sklearn.utils import shuffle
from PIL import Image
from pillow_heif import register_heif_opener

register_heif_opener()

dictLetras = {
    'a': 0,
    'b': 1,
    'c': 2,
    'd': 3,
    'e': 4,
    'f': 5,
    'g': 6,
    'h': 7,
    'i': 8,
    'l': 9,
    'm': 10,
    'n': 11,
    'o': 12,
    'p': 13,
    'r': 14,
    's': 15,
    't': 16,
    'u': 17,
    'v': 18,
    'w': 19,
    'y': 20,
    'z': 21
    }


# Cargar una imagen, convertir a escala de grises,
# redimensionar y retornar la imagen como array
def getImage(image_path, height=120, width=120):
  img = cv.imread(image_path)
  img = cv.resize(img,(height, width))
  return img


# Function: buildDataset
# Parameters:
#   - DATASET_DIR (string): ruta donde están las carpetas de los datasets
#   - dictID (dict): contiene el ID de categoría en el dataset
#   - width (int: 256): ancho en pixeles de imagen al cambiar su tamaño
#   - height (int: 256): alto en pixeles de imagen al cambiar su tamaño
#   - seed (int): Determina la generación de números aleatorios para 
#     barajar los datos
# Returns:
#     Tupla que contiene dos arreglos de Numpy, uno con las imágenes y 
#     otro con las etiquetas respectiva de cada imagen.
def buildDataset(DATASET_DIR, dictID, width = 256, height = 256, seed = 123):
  labels = []
  images = []
  
  subdirs = [subdir.name for 
             subdir in os.scandir(DATASET_DIR)
             if subdir.is_dir()]
  
  for subdir in subdirs:
    SECTION_DIR = join(DATASET_DIR, subdir)
    archivos = os.listdir(SECTION_DIR)
    
    if not archivos:
        continue

    for archivo in archivos:
      if archivo.lower().split('.')[-1] == 'jpg':
        labels.append(dictID[subdir])
        img = getImage(join(SECTION_DIR, archivo), height=height, width=width)
        images.append(img)
      elif archivo.lower().split('.')[-1] == 'heic':
        labels.append(dictID[subdir])
        img = Image.open(join(SECTION_DIR, archivo)).convert('L').resize((width, height))
        img = np.asarray(img)
        images.append(img)
  
  X = np.array(images)
  y = np.array(labels)

  X, y = shuffle(X, y, random_state=seed)
  
  return X, y