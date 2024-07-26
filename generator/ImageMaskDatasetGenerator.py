# Copyright 2024 antillia.com Toshiyuki Arai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# 2024/07/26
# ImageMaskDatasetGenerator.py

import os
import sys
import nibabel as nib
import shutil
import cv2
import pydicom

import glob
import numpy as np
import math
from scipy.ndimage.interpolation import map_coordinates

from scipy.ndimage.filters import gaussian_filter

import traceback

class ImageMaskDatasetGenerator:

  def __init__(self, size=512,
              exclude_empty_mask=True,
              mask_rotation = cv2.ROTATE_90_CLOCKWISE,
              image_normalize=True, augmentation=False, debug=True):
    self.seed = 137
    self.W = size
    self.H = size
    
    self.mask_factor = 255
    self.mask_rotation= mask_rotation
    self.exclude_empty_mask = exclude_empty_mask
    self.image_normalize   = image_normalize
    self.augmentation = augmentation
    self.debug = debug
    self.blended_dir = "./blended"
    if self.debug:
      if os.path.exists(self.blended_dir):
        shutil.rmtree(self.blended_dir)
      os.makedirs(self.blended_dir)
    if self.augmentation:
      self.hflip    = True
      self.vflip    = True
      self.rotation = True
      self.ANGLES   = [90, 180, 270]

      self.deformation=True
      self.alpha    = 1300
      self.sigmoids = [8, 10,]
          
      self.distortion=True
      self.gaussina_filer_rsigma = 40
      self.gaussina_filer_sigma  = 0.5
      self.distortions           = [0.02, 0.03,]
      self.rsigma = "sigma"  + str(self.gaussina_filer_rsigma)
      self.sigma  = "rsigma" + str(self.gaussina_filer_sigma)
      
      self.resize = False
      self.resize_ratios = [0.7, 0.8, 0.9]

      self.barrel_distortion = True
      self.radius     = 0.3
      self.amounts    = [0.3]
      self.centers    = [(0.3, 0.3), (0.7, 0.3), (0.5, 0.5), (0.3, 0.7), (0.7, 0.7)]

      self.pincushion_distortion= True
      self.pincradius  = 0.3
      self.pincamounts = [-0.3]
      self.pinccenters = [(0.3, 0.3), (0.7, 0.3), (0.5, 0.5), (0.3, 0.7), (0.7, 0.7)]

  def create_mask_files(self, mask_file, output_masks_dir):
    print("--- create_mask_files {}".format(mask_file))
    data = nib.load(mask_file)
    fdata = data.get_fdata()
    #print("--- shape {}".format(fdata.shape))
    
    # shape (256, 256, 17)
    num = fdata.shape[2]
    for i in range(num):
      self.index += 1

      mask = fdata[:,:,i]
      mask = mask * self.mask_factor
      if self.exclude_empty_mask:
        if not mask.any() >0:
          print("   Skipped an empty mask")
          continue

    
      mask = cv2.resize(mask, (self.W, self.H) )
      if self.mask_rotation >=0:
        mask = cv2.rotate(mask, self.mask_rotation)

      basename = str(self.index) +  ".jpg"
    
      filepath = os.path.join(output_masks_dir, basename)
      cv2.imwrite(filepath, mask)
      print("--- Save {}".format(filepath))
      if self.augmentation:
        self.augment(mask, basename, output_masks_dir, border=(0, 0, 0), mask=True)
    return num
  
  def create_image_files(self, image_file, output_images_dir, output_masks_dir):
    print("--- create_image_files {}".format(image_file))
    data = pydicom.dcmread(image_file)
    image = data.pixel_array
        
    image = cv2.resize(image, (self.W, self.H) )
    #print("--- image shape {}".format(image.shape))
    if self.image_normalize:
      image = self.normalize(image).astype('uint8')

    basename = str(self.index) + ".jpg"
    mask_filepath = os.path.join(output_masks_dir, basename)
    filepath = os.path.join(output_images_dir, basename)

    if os.path.exists(mask_filepath):
      cv2.imwrite(filepath, image)
      print("--- Saved {}".format(filepath))
      if self.debug:
        mask = cv2.imread(mask_filepath)
        #mask = np.expand_dims(mask, axis=-1)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        print("-- image shape {}".format(image.shape))
        print("--- mask shape {}".format(mask.shape))
        image += mask
        
        blended_filepath = os.path.join(self.blended_dir, basename)
        cv2.imwrite(blended_filepath, image)
    else:
      print("   Skipped {}".format(filepath))

    if self.augmentation:
      self.augment(image, basename, output_images_dir, border=(0, 0, 0), mask=False)
  
  def normalize(self, image):
    min = np.min(image)/255.0
    max = np.max(image)/255.0
    scale = (max - min)
    image = (image - min) / scale
    image = image.astype(np.uint32) 
    return image   

  def generate(self, input_images_dir, input_masks_dir, 
                        output_images_dir, output_masks_dir):

    image_files = glob.glob(input_images_dir + "/*.dcm")
    mask_files  = glob.glob(input_masks_dir)
    image_files = sorted(image_files)
    mask_files  = sorted(mask_files)
    num_image_files = len(image_files)
    num_mask_files  = len(mask_files)
    print("--- num_image_files {}".format(num_image_files))
    print("--- num_mask_files  {}".format(num_mask_files))

    self.index = 10000
    
    for mask_file in mask_files:
      self.create_mask_files(mask_file,   output_masks_dir)

    self.index = 10000

    for image_file in image_files:
      self.index += 1
      self.create_image_files(image_file, output_images_dir, output_masks_dir)
      
      
  def resize_to_square(self, image):
      
    h, w = image.shape[:2]
    RESIZE = h
    if w > h:
      RESIZE = w
    # 1. Create a black background
    background = np.zeros((RESIZE, RESIZE, 3),  np.uint8) 
    x = int((RESIZE - w)/2)
    y = int((RESIZE - h)/2)
    # 2. Paste the image to the background 
    background[y:y+h, x:x+w] = image
    # 3. Resize the background to (512x512)
    resized = cv2.resize(background, (self.W, self.H))

    return resized

  def augment(self, image, basename, output_dir, border=(0, 0, 0), mask=False):
    border = image[2][2].tolist()
  
    print("---- border {}".format(border))
    if self.hflip:
      flipped = self.horizontal_flip(image)
      output_filepath = os.path.join(output_dir, "hflipped_" + basename)
      cv2.imwrite(output_filepath, flipped)
      print("--- Saved {}".format(output_filepath))

    if self.vflip:
      flipped = self.vertical_flip(image)
      output_filepath = os.path.join(output_dir, "vflipped_" + basename)
      cv2.imwrite(output_filepath, flipped)
      print("--- Saved {}".format(output_filepath))

    if self.rotation:
      self.rotate(image, basename, output_dir, border)

    if self.deformation:
      self.deform(image, basename, output_dir)

    if self.distortion:
      self.distort(image, basename, output_dir)

    if self.resize:
      self.shrink(image, basename, output_dir, mask)

    if self.barrel_distortion:
      self.barrel_distort(image, basename, output_dir)


  def horizontal_flip(self, image): 
    print("shape image {}".format(image.shape))
    if len(image.shape)==3:
      return  image[:, ::-1, :]
    else:
      return  image[:, ::-1, ]

  def vertical_flip(self, image):
    if len(image.shape) == 3:
      return image[::-1, :, :]
    else:
      return image[::-1, :, ]

  def rotate(self, image, basename, output_dir, border):
    for angle in self.ANGLES:      
      center = (self.W/2, self.H/2)
      rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1)

      rotated_image = cv2.warpAffine(src=image, M=rotate_matrix, dsize=(self.W, self.H), borderValue=border)
      output_filepath = os.path.join(output_dir, "rotated_" + str(angle) + "_" + basename)
      cv2.imwrite(output_filepath, rotated_image)
      print("--- Saved {}".format(output_filepath))

  def deform(self, image, basename, output_dir): 
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    random_state = np.random.RandomState(self.seed)

    shape = image.shape
    for sigmoid in self.sigmoids:
      dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigmoid, mode="constant", cval=0) * self.alpha
      dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigmoid, mode="constant", cval=0) * self.alpha
      #dz = np.zeros_like(dx)

      x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
      indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

      deformed_image = map_coordinates(image, indices, order=1, mode='nearest')  
      deformed_image = deformed_image.reshape(image.shape)

      image_filename = "deformed" + "_alpha_" + str(self.alpha) + "_sigmoid_" +str(sigmoid) + "_" + basename
      image_filepath  = os.path.join(output_dir, image_filename)
      cv2.imwrite(image_filepath, deformed_image)

  # This method is based on the code of the following stackoverflow.com webstie:
  # https://stackoverflow.com/questions/41703210/inverting-a-real-valued-index-grid/78031420#78031420
  def distort(self, image, basename, output_dir):
    shape = (image.shape[1], image.shape[0])
    (w, h) = shape
    xsize = w
    if h>w:
      xsize = h
    # Resize original img to a square image
    resized = cv2.resize(image, (xsize, xsize))
    shape   = (xsize, xsize)
 
    t = np.random.normal(size = shape)
    for size in self.distortions:
      filename = "distorted_" + str(size) + "_" + self.sigma + "_" + self.rsigma + "_" + basename
      output_file = os.path.join(output_dir, filename)    
      dx = gaussian_filter(t, self.gaussina_filer_rsigma, order =(0,1))
      dy = gaussian_filter(t, self.gaussina_filer_rsigma, order =(1,0))
      sizex = int(xsize*size)
      sizey = int(xsize*size)
      dx *= sizex/dx.max()
      dy *= sizey/dy.max()

      image = gaussian_filter(image, self.gaussina_filer_sigma)

      yy, xx = np.indices(shape)
      xmap = (xx-dx).astype(np.float32)
      ymap = (yy-dy).astype(np.float32)

      distorted = cv2.remap(resized, xmap, ymap, cv2.INTER_LINEAR)
      distorted = cv2.resize(distorted, (w, h))
      cv2.imwrite(output_file, distorted)
      print("=== Saved distorted image file{}".format(output_file))

  def shrink(self, image, basename, output_dir, mask):
    print("----shrink shape {}".format(image.shape))
    h, w    = image.shape[0:2]
    pixel   = image[2][2]
    for resize_ratio in self.resize_ratios:
      rh = int(h * resize_ratio)
      rw = int(w * resize_ratio)
      resized = cv2.resize(image, (rw, rh))
      h1, w1  = resized.shape[:2]
      y = int((h - h1)/2)
      x = int((w - w1)/2)
      # black background
      background = np.zeros((w, h, 3), np.uint8)
      if mask == False:
        # white background
        background = np.ones((h, w, 3), np.uint8) * pixel
      # paste resized to background
      print("---shrink mask {} rsized.shape {}".format(mask, resized.shape))
      background[x:x+w1, y:y+h1] = resized
      filename = "shrinked_" + str(resize_ratio) + "_" + basename
      output_file = os.path.join(output_dir, filename)    

      cv2.imwrite(output_file, background)
      print("=== Saved shrinked image file{}".format(output_file))

  # This method is based on the code in the following stackoverflow.com website:
  # https://stackoverflow.com/questions/59776772/python-opencv-how-to-apply-radial-barrel-distortion
  def barrel_distort(self, image, basename, output_dir):    
    (h,  w,  _) = image.shape

    # set up the x and y maps as float32
    map_x = np.zeros((h, w), np.float32)
    map_y = np.zeros((h, w), np.float32)

    scale_x = 1
    scale_y = 1
    index   = 1000
    for amount in self.amounts:
      for center in self.centers:
        index += 1
        (ox, oy) = center
        center_x = w * ox
        center_y = h * oy
        radius = w * self.radius
           
        # negative values produce pincushion
 
        # create map with the barrel pincushion distortion formula
        for y in range(h):
          delta_y = scale_y * (y - center_y)
          for x in range(w):
            # determine if pixel is within an ellipse
            delta_x = scale_x * (x - center_x)
            distance = delta_x * delta_x + delta_y * delta_y
            if distance >= (radius * radius):
              map_x[y, x] = x
              map_y[y, x] = y
            else:
              factor = 1.0
              if distance > 0.0:
                v = math.sqrt(distance)
                factor = math.pow(math.sin(math.pi * math.sqrt(distance) / radius / 2), amount)
              map_x[y, x] = factor * delta_x / scale_x + center_x
              map_y[y, x] = factor * delta_y / scale_y + center_y
            
        # do the remap
        image = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
        filename = "barrdistorted_"+str(index) + "_" + str(self.radius) + "_" + str(amount) + "_" + basename
        output_filepath = os.path.join(output_dir, filename)
        cv2.imwrite(output_filepath, image)
  
  # This method is based on the code in the following stackoverflow.com website:
  # https://stackoverflow.com/questions/59776772/python-opencv-how-to-apply-radial-barrel-distortion
  def pincushion_distort(self, image, basename, output_dir):    
    (h,  w,  _) = image.shape

    # set up the x and y maps as float32
    map_x = np.zeros((h, w), np.float32)
    map_y = np.zeros((h, w), np.float32)

    scale_x = 1
    scale_y = 1
    index   = 1000
    for amount in self.pincamounts:
      for center in self.pinccenters:
        index += 1
        (ox, oy) = center
        center_x = w * ox
        center_y = h * oy
        radius = w * self.pincradius
           
        # negative values produce pincushion

        # create map with the barrel pincushion distortion formula
        for y in range(h):
          delta_y = scale_y * (y - center_y)
          for x in range(w):
            # determine if pixel is within an ellipse
            delta_x = scale_x * (x - center_x)
            distance = delta_x * delta_x + delta_y * delta_y
            if distance >= (radius * radius):
              map_x[y, x] = x
              map_y[y, x] = y
            else:
              factor = 1.0
              if distance > 0.0:
                v = math.sqrt(distance)
                factor = math.pow(math.sin(math.pi * math.sqrt(distance) / radius / 2), amount)
              map_x[y, x] = factor * delta_x / scale_x + center_x
              map_y[y, x] = factor * delta_y / scale_y + center_y
            
        # do the remap
        image = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
        filename = "pincdistorted_"+str(index) + "_" + str(self.pincradius) + "_" + str(amount) + "_" + basename
        output_filepath = os.path.join(output_dir, filename)
        cv2.imwrite(output_filepath, image)
  

if __name__ == "__main__":
  try:
    #type = "CE-T1"
    #type = "T1"
    type = "T2"
    if len(sys.argv) == 2:
      type = sys.argv[1]
    if not (type == "T1" or type == "T2" or type=="CE-T1"):
      error = "Invalid type: specify T1, T2 or CE-T1"
      raise Exception(error)
    
    input_images_dir  = "./MRI-Segments/*/" + type + "WI/"  #*.dcm
    #                      MRI-Segments/001/T2WI/

    input_masks_dir   = "./MRI-Segments/*/ROI-"+ type + ".nii"
    output_images_dir = "./Nasopharyngeal-Carcinoma-" + type + "WI/images/"
    output_masks_dir  = "./Nasopharyngeal-Carcinoma-" + type + "WI/masks/"

    if os.path.exists(output_images_dir):
      shutil.rmtree(output_images_dir)
    if not os.path.exists(output_images_dir):
      os.makedirs(output_images_dir)

    if os.path.exists(output_masks_dir):
      shutil.rmtree(output_masks_dir)
    if not os.path.exists(output_masks_dir):
      os.makedirs(output_masks_dir)

    size         = 512
    # Enabled exclude_empty_mask flag
    exclude_empty_mask = True
    
    # Specified an angle to rotate a mask.
    mask_rotation   = cv2.ROTATE_90_COUNTERCLOCKWISE

    # Enabled image_normalization flag.
    image_normalize=True

    # Disabled an offline augmentatin flag.
    augmentation = False
  
    generator = ImageMaskDatasetGenerator(size = size, 
                                          exclude_empty_mask = exclude_empty_mask,
                                          mask_rotation = mask_rotation,
                                          image_normalize = image_normalize,
                                          augmentation = augmentation)
    generator.generate(input_images_dir, input_masks_dir, 
                        output_images_dir, output_masks_dir)
  except:
    traceback.print_exc()


