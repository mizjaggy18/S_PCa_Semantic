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
from skimage.filters import threshold_otsu

def generate_grade_group(gleason_score):
    """
    Map the Gleason score to its corresponding Grade Group.
    """
    # Mapping Gleason scores to Grade Groups
    grade_group_mapping = {
        "3+3": 1,
        "3+4": 2,
        "4+3": 3,
        "4+4": 4,
        "4+5": 5,
        "5+4": 5,
        "5+5": 5
    }
    
    # Check if the score is in the mapping
    if gleason_score in grade_group_mapping:
        return f"Grade Group {grade_group_mapping[gleason_score]}"
    else:
        # Handle Gleason scores not explicitly listed
        score_sum = sum(map(int, gleason_score.split('+')))
        if score_sum >= 9:
            return "Grade Group 5"
        return "Unknown Grade Group"

def generate_gleason_score(class_percentages):
    gleason_classes = {idx: class_percentages.get(idx, 0) for idx in [3, 4, 5]}

    for class_idx, percentage in gleason_classes.items():
        if percentage == 100 and all(gleason_classes[other] == 0 for other in gleason_classes if other != class_idx):
            return f"{class_idx}+{class_idx}"
    
    sorted_classes = sorted(gleason_classes.items(), key=lambda x: x[1], reverse=True)
    gleason_5_priority = gleason_classes[5] >= 5
    if gleason_5_priority:
        highest_class = sorted_classes[0][0]
        gleason_score = f"{highest_class}+5" if highest_class != 5 else "5+5"
    else:
        gleason_score = f"{sorted_classes[0][0]}+{sorted_classes[1][0]}"
    return gleason_score


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
    th_area = parameters.cytomine_area_th
    num_classes = 6
    maxsize = parameters.maxsize
    patch_size = parameters.patch_size
    patch_width = patch_size
    patch_height = patch_size
    input_size = parameters.input_size
    overlap = parameters.overlap

    terms = TermCollection().fetch_with_filter("project", parameters.cytomine_id_project)
    job.update(status=Job.RUNNING, progress=1, statusComment="Terms collected...")
    print(terms)

    start_time=time.time()
    
    # modelpath="./models/best_unet_dn21_pytable_PANDA-random-30p-1024-nonorm-pt_100.pth" ##### ***** #####
    modelpath="/models/best_unet_dn21_pytable_PANDA-random-30p-multitiles-coloraug-pt_47ep.pth" ##### ***** #####
    
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

    max_image_size = 1000
    threshold_allowance = 10
    kernel_size = 3
    image_area_perc_threshold = 0.001

    #Set working path
    working_path = os.path.join("tmp", str(job.id))
   
    if not os.path.exists(working_path):
        logging.info("Creating working directory: %s", working_path)
        os.makedirs(working_path)
    try:

        id_project=project.id   
        output_path = os.path.join(working_path, "prostate_gleason_results.csv")
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

            resize_ratio = max(wsi_width, wsi_height) / max_image_size
            if resize_ratio < 1:
                resize_ratio = 1

            resized_width = int(wsi_width / resize_ratio)
            resized_height = int(wsi_height / resize_ratio)
            dim = (resized_width, resized_height)  
            bit_depth = 8
            

            response = cyto_job.get_instance()._get(
                "{}/{}/window-{}-{}-{}-{}.{}".format("imageinstance", id_image, 0, 0, wsi_width, wsi_height, "png"), {"maxSize":max_image_size}
            )

            if response.status_code in [200, 304] and response.headers['Content-Type'] == 'image/png':
                image_bytes = BytesIO(response.content)
                image_array = np.asarray(bytearray(image_bytes.read()), dtype=np.uint8)
                im = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)
                im_resized = cv2.resize(im, dim, interpolation = cv2.INTER_AREA)
                pixels = np.array(im_resized).flatten()
                th_value = threshold_otsu(pixels)
                print("Otsu threshold: ", th_value)
                threshold = th_value + threshold_allowance
                print("Otsu threshold + allowance: ", threshold)
                thresh_mask = (im < threshold).astype(np.uint8)*255
                
                kernel_size = np.array(kernel_size)
                if kernel_size.size != 2:  
                    kernel_size = kernel_size.repeat(2)
                kernel_size = tuple(np.round(kernel_size).astype(int))
                
                # Create structuring element for morphological operations
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
                min_region_size = np.sum(kernel)
                _, output, stats, _ = cv2.connectedComponentsWithStats(thresh_mask, connectivity=8)
                sizes = stats[1:, -1]
                for i, size in enumerate(sizes):
                    if size < min_region_size:
                        thresh_mask[output == i + 1] = 0

                thresh_mask = cv2.morphologyEx(thresh_mask, cv2.MORPH_OPEN, kernel)
                thresh_mask = cv2.bitwise_not(thresh_mask)            

                extension = 10
                extended_img = cv2.copyMakeBorder(
                    thresh_mask,
                    extension,
                    extension,
                    extension,
                    extension,
                    cv2.BORDER_CONSTANT,
                    value=255  # Use the same as the background value
                )

                h, w = thresh_mask.shape
                edges = np.zeros_like(thresh_mask)
                edges[0, :] = edges[-1, :] = edges[:, 0] = edges[:, -1] = 1
                
                mask_edges = cv2.bitwise_and(thresh_mask, edges)
                thresh_mask[mask_edges > 0] = 0

                # extract foreground polygons 
                fg_objects = mask_to_objects_2d(extended_img, background=255, offset=(-extension, -extension))
                zoom_factor = wsi_width / float(resized_width)

                # Only keep components greater than {image_area_perc_threshold}% of whole image
                min_area = int((image_area_perc_threshold / 100) * wsi_width * wsi_height)
                total_tissue_area = 0
                transform_matrix = [zoom_factor, 0, 0, -zoom_factor, 0, wsi_height]
                annotations = AnnotationCollection()
                for i, (fg_poly, _) in enumerate(fg_objects):                    
                    upscaled = affine_transform(fg_poly, transform_matrix)
                    total_tissue_area += upscaled.area
                    if upscaled.area <= min_area:
                        continue
                    # print(upscaled.area)
                    try:
                        print("Mask area: ", upscaled.area)                            
                        # Annotation(
                        # location=upscaled.wkt,
                        # id_image=id_image,
                        # id_terms=[predictroi_term],
                        # id_project=cytomine_id_project).save()                    
                    except:
                        print("An exception occurred. Proceed with next annotations")
            
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
                # try:                
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
                # patch_size = 1000
                # input_size = 128
                # maxsize = 256
                # overlap = 0.5
                print(f"patch size: {patch_size}, input_size: {input_size}, maxsize: {maxsize}, overlap: {overlap}")
                step = int(patch_size * (1 - overlap))  # 50% overlap
                num_patches_x = (roi_width + step - 1) // step
                num_patches_y = (roi_height + step - 1) // step
                print(f'Patch X: {num_patches_x}, Patch Y: {num_patches_y}')
                print(f'Step X: {step}, Step Y: {step}')
     
                num_classes = 6
                segmentation_result = np.zeros((roi_height, roi_width, num_classes), dtype=np.float32)
                # overlap_count = np.zeros((roi_height, roi_width), dtype=np.float32)  # For tracking overlap per pixel
                # weight_matrix = create_weight_matrix(patch_size)

                # is_algo = User().fetch(roi.user).algo   

                for i in range(0, roi_height, step):
                    for j in range(0, roi_width, step):
                        if i + patch_size > roi_height:
                            i = roi_height - patch_size
                        if j + patch_size > roi_width:
                            j = roi_width - patch_size
                        if roi_width < patch_size:
                            j=0
                            patch_width = roi_width
                        if roi_height < patch_size:
                            i=0
                            patch_height = roi_height

                        patch_height = patch_size if roi_height >= patch_size else roi_height
                        patch_width = patch_size if roi_width >= patch_size else roi_width
                        
                        patch_x = int(min_x) + j
                        patch_y = int(wsi_height - max_y) + i

                        x, y, w, h = patch_x, patch_y, patch_width, patch_height
                        # response = cyto_job.get_instance()._get(
                        #     "{}/{}/window-{}-{}-{}-{}.{}".format("imageinstance", id_image, x, y, w, h, "png"),{})
                        response = cyto_job.get_instance()._get(
                            "{}/{}/window-{}-{}-{}-{}.{}".format("imageinstance", id_image, x, y, w, h, "png"),{"maxSize":maxsize})
                        
                        if response.status_code in [200, 304] and response.headers['Content-Type'] == 'image/png':
                            roi_im = Image.open(BytesIO(response.content))
                            gray_im = roi_im.convert("L")
                            min_pixel, max_pixel = gray_im.getextrema()
                            if min_pixel >= 250 and max_pixel >= 250:
                                continue                 

                        if roi_im.mode == 'RGBA':
                            roi_im = roi_im.convert("RGB")

                        transform = transforms.Compose([
                            transforms.Resize((input_size,input_size)), 
                            transforms.ToTensor(),
                        ])

                        image_tensor = transform(roi_im).unsqueeze(0).to(device)  # Add batch dimension [1, C, H, W]
                        # Forward pass
                        with torch.no_grad():
                            segmented_patch = model(image_tensor)  

                        # Convert model predictions to class probabilities and resize
                        seg_probs = torch.softmax(segmented_patch, dim=1).cpu().numpy()
                        seg_probs_resized = [resize(seg_probs[0, c], (patch_height, patch_width), order=1, preserve_range=True, anti_aliasing=True) for c in range(num_classes)]

                        for c in range(num_classes):
                            weighted_class_map = seg_probs_resized[c] * 1 
                            segmentation_result[i:i + patch_height, j:j + patch_width, c] += weighted_class_map
                            
                final_segmentation = np.uint8(np.argmax(segmentation_result, axis=-1))

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
                    # buffer_distance = 1.0

                    # # Collect polygons for each class
                    for i, (fg_poly, _) in enumerate(fg_objects):
                        upscaled = affine_transform(fg_poly, transform_matrix)
                        if upscaled.area >= min_area:
                            outer_boundary = Polygon(upscaled.exterior)
                            # class_polygons[class_idx]["polygons"].append(outer_boundary)
                            # smoothed_polygon = outer_boundary.buffer(buffer_distance).buffer(-buffer_distance)                            
                            class_polygons[class_idx]["polygons"].append(outer_boundary)
                            # upscaled = smoothed_polygon
                # except:
                # # finally:
                #     # print("end")
                #     print("An exception occurred. Proceed with next annotations")
            class_areas = {}

            # After processing all ROIs, combine neighboring polygons by class
            for class_idx, class_data in class_polygons.items():
                polygons = class_data["polygons"]
                id_terms = class_data["id_terms"]

                # Merge polygons for the class using unary_union
                combined_polygon = unary_union(polygons)

                total_area = 0

                # Check if combined_polygon is a MultiPolygon
                if combined_polygon.geom_type == 'MultiPolygon':
                    # Loop through each part of the MultiPolygon
                    for poly in combined_polygon.geoms:  # Use .geoms to iterate over each Polygon in the MultiPolygon
                        # Save each polygon as a separate annotation with class-specific id_terms
                        total_area += poly.area
                        Annotation(
                            location=poly.wkt,
                            id_image=id_image,
                            id_terms=[id_terms],
                            id_project=project.id
                        ).save()
                else:
                    total_area = combined_polygon.area
                    # Save the single combined polygon as an annotation
                    Annotation(
                        location=combined_polygon.wkt,
                        id_image=id_image,
                        id_terms=[id_terms],
                        id_project=project.id
                    ).save()

                class_areas[class_idx] = total_area

            # Calculate the total area across all classes
            total_area_all_classes = sum(class_areas.values())

            # Calculate the percentage area for each class
            class_percentages = {class_idx: (area / total_area_all_classes) * 100 for class_idx, area in class_areas.items()}
            class_names = {3: "Gleason 3", 4: "Gleason 4", 5: "Gleason 5"}

            print("Tissue volume:", total_tissue_area)
            print("Tumor area:",total_area_all_classes)
            tumor_volume_ratio=total_area_all_classes/total_tissue_area
            print("Tumor volume (tumor/tissue ratio):",tumor_volume_ratio)
            # print("Total area for each pattern:", class_areas)
            # print("Percentage area for each pattern:", class_percentages)

            print("Total area (percentage) for each pattern:")
            for class_idx, area in class_areas.items():
                percentage = class_percentages.get(class_idx, 0)  # Get the percentage, default to 0
                class_name = class_names.get(class_idx, f"Class {class_idx}")  # Get the class name
                print(f"{class_name}: {area:.1f} ({percentage:.2f}%)")

            gleason_score = generate_gleason_score(class_percentages)
            print("Gleason Score:", gleason_score)
            grade_group = generate_grade_group(gleason_score)
            print("Grade Group:", grade_group)
            
            end_prediction_time=time.time()
                  
            end_time=time.time()
            print("Execution time: ",end_time-start_time)

            f.write("\n")
            gleason_3_area = class_areas.get(3, 0.0)
            gleason_3_percent = class_percentages.get(3, 0.0)
            gleason_4_area = class_areas.get(4, 0.0)
            gleason_4_percent = class_percentages.get(4, 0.0)
            gleason_5_area = class_areas.get(5, 0.0)
            gleason_5_percent = class_percentages.get(5, 0.0)
            
            # Write the data
            f.write("{};{:.2f};{:.2f};{:.8f};{:.2f} ({:.2f}%);{:.2f} ({:.2f}%);{:.2f} ({:.2f}%);{};{};{:.2f}\n".format(
                id_image,
                total_tissue_area,
                total_area_all_classes,
                tumor_volume_ratio,
                gleason_3_area, gleason_3_percent,
                gleason_4_area, gleason_4_percent,
                gleason_5_area, gleason_5_percent,
                gleason_score,
                grade_group,
                end_time-start_time
            ))
                    
        f.close()
        
        job.update(status=Job.RUNNING, progress=99, statusComment="Summarizing results...")
        job_data = JobData(job.id, "Generated File", "prostate_gleason_results.csv").save()
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

