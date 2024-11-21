# -*- coding: utf-8 -*-

# * Copyright (c) 2009-2018. Authors: see NOTICE file.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *      http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.

from __future__ import print_function, unicode_literals, absolute_import, division
import sys
import numpy as np
# from pathlib import Path
import os
import cytomine
from shapely.geometry import shape, box, Polygon, Point, MultiPolygon
from shapely import wkt
from shapely.ops import unary_union
from shapely.affinity import affine_transform
from glob import glob
from sldc.locator import mask_to_objects_2d

from tifffile import imread
from cytomine import Cytomine, models, CytomineJob
from cytomine.models import Annotation, AnnotationTerm, AnnotationCollection, ImageInstanceCollection, Job, User, JobData, Project, ImageInstance, Property
from cytomine.models.ontology import Ontology, OntologyCollection, Term, RelationTerm, TermCollection

# from csbdeep.utils import Path, normalize
# from stardist.models import StarDist2D

import torch
from torchvision.models import DenseNet
import openvino as ov

# import tensorflow as tf
# from tensorflow import keras

from PIL import Image
from skimage import io, color, filters, measure
from scipy.ndimage import zoom

# import matplotlib.pyplot as plt
import time
import cv2
import math
import csv

from argparse import ArgumentParser
import json
import logging
import logging.handlers
import shutil
import tempfile
import requests
from io import BytesIO
from skimage.transform import resize

__author__ = "WSH Munirah W Ahmad <wshmunirah@gmail.com>"
__version__ = "1.0.1"
# Semantic Segmentation using custom encoder for Prostate Cancer (Date created: 17 Sep 2024)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import DenseNet
from torchvision import transforms
from scipy.ndimage import gaussian_filter
from scipy.ndimage import median_filter

def create_weight_matrix(patch_size, sigma=0.5):
    # Initialize a matrix with ones (center gets more weight)
    center_matrix = np.ones((patch_size, patch_size), dtype=np.float32)
    # Apply Gaussian filter to create the weight matrix
    weight_matrix = gaussian_filter(center_matrix, sigma=patch_size * sigma)
    # Normalize weights so that the center values are higher
    weight_matrix /= weight_matrix.max()
    return weight_matrix

class UNetWithDenseNetEncoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=6):
        super(UNetWithDenseNetEncoder, self).__init__()

        # Use DenseNet121 as the encoder
        densenet = DenseNet(growth_rate=32, block_config=(2, 2, 2, 2), 
                            num_init_features=64, bn_size=4, drop_rate=0)

        # Extract the features (remove the classifier part)
        self.encoder = densenet.features

        # Number of channels after each dense block based on the actual architecture
        self.enc_channels = [64, 128, 64, 128, 64]  # Adjusted for DenseNet growth_rate and block config

        # Decoder (Upsampling) - adjusted channels to match encoder output
        self.up1 = nn.ConvTranspose2d(self.enc_channels[4], self.enc_channels[3], kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(self.enc_channels[3] + self.enc_channels[3], self.enc_channels[3], kernel_size=3, padding=1)

        self.up2 = nn.ConvTranspose2d(self.enc_channels[3], self.enc_channels[2], kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(self.enc_channels[2] + self.enc_channels[2], self.enc_channels[2], kernel_size=3, padding=1)

        self.up3 = nn.ConvTranspose2d(self.enc_channels[2], self.enc_channels[1], kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(self.enc_channels[1] + self.enc_channels[1], self.enc_channels[1], kernel_size=3, padding=1)

        self.up4 = nn.ConvTranspose2d(self.enc_channels[1], self.enc_channels[0], kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(self.enc_channels[0] + self.enc_channels[0], self.enc_channels[0], kernel_size=3, padding=1)

        # Final output layer
        self.final = nn.Conv2d(self.enc_channels[0], out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder: DenseNet's features
        enc0 = self.encoder[0](x)  # First convolution
        enc1 = self.encoder[4](enc0)  # Block 1
        enc2 = self.encoder[5](enc1)  # Block 2
        enc3 = self.encoder[6](enc2)  # Block 3
        enc4 = self.encoder[7](enc3)  # Block 4 (final DenseNet block)

        # Decoder with skip connections
        x = self.up1(enc4)  # Upsample Block 4 to Block 3 size
        x = torch.cat([x, enc3], dim=1)  # Concatenate with corresponding encoder feature map
        x = F.relu(self.conv1(x))  # Conv1 expects 512 + 512 channels
        x = self.up2(x)  # Upsample Block 3 to Block 2 size
        
        # Ensure x and enc2 have matching spatial dimensions
        x = F.interpolate(x, size=enc2.shape[2:], mode='bilinear', align_corners=False)
        
        x = torch.cat([x, enc2], dim=1)  # Concatenate with corresponding encoder feature map
        x = F.relu(self.conv2(x))  # Conv2 expects 256 + 256 channels
        x = self.up3(x)  # Upsample Block 2 to Block 1 size
        
        # Ensure x and enc1 have matching spatial dimensions
        x = F.interpolate(x, size=enc1.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, enc1], dim=1)  # Concatenate with corresponding encoder feature map
        x = F.relu(self.conv3(x))  # Conv3 expects 128 + 128 channels
        x = self.up4(x)  # Upsample Block 1 to original size
        
        # Ensure x and enc0 have matching spatial dimensions
        x = F.interpolate(x, size=enc0.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, enc0], dim=1)  # Concatenate with first feature map
        x = F.relu(self.conv4(x))  # Conv4 expects 64 + 64 channels
        return self.final(x)


def run(cyto_job, parameters):
    logging.info("----- PCa-Semantic-UNet-DenseNet v%s -----", __version__)
    logging.info("Entering run(cyto_job=%s, parameters=%s)", cyto_job, parameters)

    job = cyto_job.job
    user = job.userJob
    project = cyto_job.project
    # th_area = parameters.cytomine_area_th
    num_classes = 6
    colour_transform = parameters.colour_transform
    patch_size = parameters.patch_size
    overlap = parameters.overlap
    
    

    terms = TermCollection().fetch_with_filter("project", parameters.cytomine_id_project)
    job.update(status=Job.RUNNING, progress=1, statusComment="Terms collected...")
    print(terms)

    start_time=time.time()

    modelpath="/models/best_unet_dn21_pytable_PANDA-random-30p-1024-nonorm-pt_100.pth" ##### ***** #####
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = UNetWithDenseNetEncoder(in_channels=3, out_channels=6).to(device)
    # model.load_state_dict(torch.load(modelpath, map_location=torch.device('cpu')))
    model.load_state_dict(torch.load(modelpath, map_location=device))
    model.eval()  # Set the model to evaluation mode
    model.to(device)
    
    # ------------------------

    print("Model successfully loaded!")
    job.update(status=Job.RUNNING, progress=20, statusComment=f"Model successfully loaded!")

    #Select images to process
    images = ImageInstanceCollection().fetch_with_filter("project", project.id)       
    list_imgs = []
    if parameters.cytomine_id_images == 'all':
        for image in images:
            list_imgs.append(int(image.id))
    else:
        list_imgs = parameters.cytomine_id_images
        list_imgs2 = list_imgs.split(',')
        
    print('Print list images:', list_imgs2)
    job.update(status=Job.RUNNING, progress=30, statusComment="Images gathered...")

    #Set working path
    working_path = os.path.join("tmp", str(job.id))
   
    if not os.path.exists(working_path):
        logging.info("Creating working directory: %s", working_path)
        os.makedirs(working_path)
    try:

        id_project=project.id   
        output_path = os.path.join(working_path, "classification_results.csv")
        f= open(output_path,"w+")

        f.write("AnnotationID;ImageID;ProjectID;JobID;TermID;UserID;Area;Perimeter;Hue;Value;WKT \n")
        
        #Go over images
        for id_image in list_imgs2:
            job.update(status=Job.RUNNING, progress=50, statusComment=f'Running prediction on image: {id_image}')
            print('Current image:', id_image)
            imageinfo=ImageInstance(id=id_image,project=parameters.cytomine_id_project)
            imageinfo.fetch()
            calibration_factor=imageinfo.resolution
            wsi_width=imageinfo.width
            wsi_height=imageinfo.height
            
            #term for large ROI
            roi_annotations = AnnotationCollection(
                terms=[parameters.cytomine_id_roi_term],
                project=parameters.cytomine_id_project,
                image=id_image, #conn.parameters.cytomine_id_image
                showWKT = True,
                includeAlgo=True, 
            )
            roi_annotations.fetch()
            print(roi_annotations)

            class_polygons = {}

            #Go over ROI in this image
            #for roi in conn.monitor(roi_annotations, prefix="Running detection on ROI", period=0.1):
            for roi in roi_annotations:
                job.update(status=Job.RUNNING, progress=50, statusComment=f'Running prediction on image: {id_image}, ROI: {roi.id}')
                try:                
                    # print(".", sep=' ', end='', flush=True)
                    roi_geometry = wkt.loads(roi.location)
                    # print("ROI Geometry from Shapely: {}".format(roi_geometry))
                    min_x=roi_geometry.bounds[0]
                    min_y=roi_geometry.bounds[1]
                    max_x=roi_geometry.bounds[2]
                    max_y=roi_geometry.bounds[3]
                    roi_width=round(max_x - min_x)
                    roi_height=round(max_y - min_y)
                    print("ROI width = ", roi_width, "; ROI height = ", roi_height)
    
                    annotation = Annotation().fetch(roi.id)
                    # x, y, width, height = annotation.bbox() 
                    image_instance = ImageInstance().fetch(id_image)
                    print(image_instance)
                    print(annotation)
    
                    # Parameters
                    step = int(patch_size * (1 - overlap))  # 50% overlap
                    num_patches_x = (roi_width + step - 1) // step
                    num_patches_y = (roi_height + step - 1) // step
                    print(f'Patch X: {num_patches_x}, Patch Y: {num_patches_y}')
                    print(f'Step X: {step}, Step Y: {step}')
         
                    num_classes = 6
                    # Assume `num_classes` is the total number of segmentation classes
                    segmentation_result = np.zeros((roi_height, roi_width, num_classes), dtype=np.float32)
                    overlap_count = np.zeros((roi_height, roi_width), dtype=np.float32)  # For tracking overlap per pixel
                    class_counts = np.zeros((roi_height, roi_width, num_classes), dtype=np.int32)
    
    
                    # Loop through patches with adjustments for the last patches
                    for i in range(0, roi_height - patch_size + 1, step):
                        for j in range(0, roi_width - patch_size + 1, step):                        
                            # Adjust i and j for the last patch in each direction to fit within ROI boundaries
                            if i + patch_size > roi_height:
                                i = roi_height - patch_size
                            if j + patch_size > roi_width:
                                j = roi_width - patch_size
                            # Calculate the coordinates in the whole-slide image (WSI) system
                            patch_x = int(min_x) + j
                            patch_y = int(wsi_height - max_y) + i
    
                            # image_id = id_image
                            x, y, w, h = patch_x, patch_y, patch_size, patch_size
                            response = cyto_job.get_instance()._get(
                                "{}/{}/window-{}-{}-{}-{}.{}".format("imageinstance", id_image, x, y, w, h, "png"),{"maxSize": 256})
                            
                            if response.status_code in [200, 304] and response.headers['Content-Type'] == 'image/png':
                                roi_im = Image.open(BytesIO(response.content))
                                gray_im = roi_im.convert("L")
                                min_pixel, max_pixel = gray_im.getextrema()
                                if min_pixel == 255 and max_pixel == 255:
                                    continue               
    
                            if roi_im.mode == 'RGBA':
                                roi_im = roi_im.convert("RGB")
    
                            if colour_transform == 1:
                                import torchvision.transforms.functional as Ft
                                saturation_factor = 0.50  # Equivalent to -15% saturation
                                hue_factor = 0.07  # Fixed hue adjustment
                                transformed_img = Ft.adjust_saturation(roi_im, saturation_factor)
                                transformed_img = Ft.adjust_hue(transformed_img, hue_factor)
                            else:
                                transformed_img=roi_im
    
                            # Preprocessing: assuming the image is a PIL image, apply necessary transformations
                            transform = transforms.Compose([
                                transforms.Resize((256, 256)),  # Resize to the input size expected by the model
                                transforms.ToTensor(),          # Convert the image to a PyTorch tensor
                            ])
    
                            # Transform the input image and add a batch dimension                        
                            image_tensor = transform(transformed_img).unsqueeze(0).to(device)  # Add batch dimension [1, C, H, W]
                            # Forward pass
                            with torch.no_grad():
                                segmented_patch = model(image_tensor)  # Model output shape [1, num_classes, height, width]
                                seg_preds = torch.argmax(segmented_patch, dim=1).cpu().numpy()[0]  # Shape: [patch_size, patch_size]
    
    
                            # # Convert model predictions to class probabilities and resize
                            # Resize the predictions for alignment with patch size
                            seg_preds_resized = resize(
                                seg_preds, (patch_size, patch_size), order=0, preserve_range=True, anti_aliasing=True
                            ).astype(np.int32)  # Ensure integer class labels after resizing
    
    
                            # Accumulate class counts for each pixel in the ROI
                            for c in range(num_classes):
                                # Create a binary mask for where the class `c` was predicted
                                class_mask = (seg_preds_resized == c).astype(np.int32)
                                # Increment counts at each pixel location for the predicted class
                                class_counts[i:i + patch_size, j:j + patch_size, c] += class_mask
    
                    # Final segmentation by selecting the class with the highest count (majority vote) at each pixel
                    final_segmentation = np.uint8(np.argmax(class_counts, axis=-1))  # Shape: [roi_height, roi_width]
                    final_segmentation = median_filter(final_segmentation, size=3)
                    print(np.max(final_segmentation))
                    print(final_segmentation.shape)
    
                    # Zoom factor for WSI
                    bit_depth = 8 #imageinfo.bitDepth if imageinfo.bitDepth is not None else 8
                    zoom_factor = 1
                    transform_matrix = [zoom_factor, 0, 0, -zoom_factor, min_x, max_y]
                    min_area = int((0.0001 / 100) * wsi_width * wsi_height)
                    # print(min_area)
                    cytomine_annotations = AnnotationCollection()
    
                    job.update(status=Job.RUNNING, progress=50, statusComment="Uploading annotations...")
    
                    # Assuming that each class is represented as polygons in the segmentation output
                    for class_idx in np.unique(final_segmentation):
                        if class_idx==0:
                            continue  # Skip background class
                        elif class_idx==1:
                            # print("Class 1: Stroma")
                            # id_terms=parameters.cytomine_id_c1_term
                            continue
                        elif class_idx==2:
                            # print("Class 2: Benign")
                            # id_terms=parameters.cytomine_id_c2_term
                            continue
                        elif class_idx==3:
                            # print("Class 3: Gleason3")
                            id_terms=parameters.cytomine_id_c3_term
                        elif class_idx==4:
                            # print("Class 4: Gleason4")
                            id_terms=parameters.cytomine_id_c4_term
                        elif class_idx==5:
                            # print("Class 5: Gleason5")
                            id_terms=parameters.cytomine_id_c5_term
    
                        # Initialize the class in the dictionary if not already present
                        if class_idx not in class_polygons:
                            class_polygons[class_idx] = {
                                "polygons": [],
                                "id_terms": id_terms
                            }
    
                        # Create a binary mask for the current class
                        # class_mask = seg_preds_resized == class_idx
                        class_mask = final_segmentation == class_idx
                        fg_objects = mask_to_objects_2d(class_mask)
                        buffer_distance = 50.0
    
                        # # Collect polygons for each class
                        for i, (fg_poly, _) in enumerate(fg_objects):
                            upscaled = affine_transform(fg_poly, transform_matrix)
                            if upscaled.area >= min_area:
                                outer_boundary = Polygon(upscaled.exterior)
                                # class_polygons[class_idx]["polygons"].append(outer_boundary)
                                smoothed_polygon = outer_boundary.buffer(buffer_distance).buffer(-buffer_distance)                            
                                class_polygons[class_idx]["polygons"].append(smoothed_polygon)
                                upscaled = smoothed_polygon
                                # Annotation(
                                # location=upscaled.wkt,
                                # id_image=id_image,
                                # id_terms=[id_terms],
                                # id_project=project.id
                                # ).save()

                except:
                # finally:
                    # print("end")
                    print("An exception occurred. Proceed with next annotations")
                    
            class_areas = {}

            
            # Combine overlapping polygons by class
            for class_idx, class_data in class_polygons.items():
                polygons = class_data["polygons"]
                id_terms = class_data["id_terms"]

                # Use unary_union to merge overlapping polygons for the current class
                combined_polygon = unary_union(polygons)

                # Ensure the output is a MultiPolygon or Polygon
                if combined_polygon.geom_type == 'Polygon':
                    combined_polygon = [combined_polygon]  # Convert to list for consistency
                elif combined_polygon.geom_type == 'MultiPolygon':
                    combined_polygon = list(combined_polygon.geoms)  # Extract individual polygons

                # Save each merged polygon
                for merged_poly in combined_polygon:
                    if merged_poly.area >= min_area:  # Filter based on the minimum area
                        Annotation(
                            location=merged_poly.wkt,
                            id_image=id_image,
                            id_terms=[id_terms],
                            id_project=project.id
                        ).save()

                # Calculate the total area for this class
                total_area = sum(poly.area for poly in combined_polygon)
                class_areas[class_idx] = total_area

            # Calculate the total area across all classes
            total_area_all_classes = sum(class_areas.values())

            # Calculate the percentage area for each class
            class_percentages = {class_idx: (area / total_area_all_classes) * 100 for class_idx, area in class_areas.items()}

            # Print or process the results as needed
            print("Total area for each class:", class_areas)
            print("Percentage area for each class:", class_percentages)

            end_prediction_time=time.time()
                  
            end_time=time.time()
            print("Execution time: ",end_time-start_time)
            # print("Prediction time: ",end_prediction_time-start_prediction_time)

            f.write("\n")
            f.write("Image ID;Class Prediction;Class 0 (Others);Class 1 (Necrotic);Class 2 (Tumor);Total Prediction;Execution Time;Prediction Time\n")
            # f.write("{};{};{};{};{};{};{};{}\n".format(id_image,im_pred,pred_c0,pred_c1,pred_c2,pred_total,end_time-start_time,end_prediction_time-start_prediction_time))
            
        f.close()
        
        job.update(status=Job.RUNNING, progress=99, statusComment="Summarizing results...")
        job_data = JobData(job.id, "Generated File", "classification_results.csv").save()
        job_data.upload(output_path)

    finally:
        logging.info("Deleting folder %s", working_path)
        shutil.rmtree(working_path, ignore_errors=True)
        logging.debug("Leaving run()")


    job.update(status=Job.TERMINATED, progress=100, statusComment="Finished.") 

if __name__ == "__main__":
    logging.debug("Command: %s", sys.argv)

    with cytomine.CytomineJob.from_cli(sys.argv) as cyto_job:
        run(cyto_job, cyto_job.parameters)

