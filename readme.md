# Airbnb's Amenity Detection with Detectron2

- [Airbnb's Amenity Detection with Detectron2](#airbnbs-amenity-detection-with-detectron2)
  - [Overview](#overview)
  - [Dataset description](#dataset-description)
  - [Data Prepartion](#data-prepartion)
  - [Register Custom Datasets](#register-custom-datasets)
  - [References:](#references)

## Overview



## Dataset description

Google's public available Open Images V5 has been used which contains 15.4M annotated bounding boxes for over 600 object categories. It has 1.9M images and is largest among all existing datasets with object location annotations. The classes include a variety of objects in various categories. It covers classes varying from different kinds of musical instruments(e.g. organ, cello, piano etc.) to different kinds of aquatic animals(e.g. goldfish, crab, seahorse, oyster etc.) to various kinds of kitchenware(e.g. spoon, kitchen knife, frying pan, dishwasher) and so on.

Since our focus is on animitey detection, I filtered out images corresponding to objects that don't fall in this category. Below are choosen subset of classes that relate most to Airnb business.

```py
{
'Bathtub',
 'Bed',
 'Billiard table',
 'Ceiling fan',
 'Coffeemaker',
 'Couch',
 'Countertop',
 'Dishwasher',
 'Fireplace',
 'Fountain',
 'Gas stove',
 'Jacuzzi',
 'Kitchen & dining room table',
 'Microwave oven',
 'Mirror',
 'Oven',
 'Pillow',
 'Porch',
 'Refrigerator',
 'Shower',
 'Sink',
 'Sofa bed',
 'Stairs',
 'Swimming pool',
 'Television',
 'Toilet',
 'Towel',
 'Tree house',
 'Washing machine',
 'Wine rack'
}
```

This article nicely expalins download procedure in detail 
https://www.learnopencv.com/fast-image-downloader-for-open-images-v4/

## Data Prepartion

Finally, Downloaded Raw Dataset from open images contains following files:

``` yaml
Images: Raw images are placed in the folder. images are named under unique ids as 'id.jpg'.
  train- images folder for training model
  validation- images folder for model validation

Annotations-file: annotations are provided in csv files with image_id, object class_id, coordinates of annotation boxes that corresponding to object.
  - train-annotations-bbox.csv
  - validation-annotations-bbox.csv

class-description: contains mapping info class_id to object name.
  class-descriptions-boxable.csv
```

Since we are using **Detectron** model, labels comprising category and bounding box are in csv format. But Detectron expect labels in JSON format in [COCO style](https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch). For that ```modules/preprocessing.py``` script is used. 

Below is sample label   
``` json
{"file_name": "train/cmaker-bathtub-treehouse-train/e43f28c69c3bb136.jpg", 
"image_id": 55, 
"height": 1024, 
"width": 732, 
"annotations": [{"bbox": [0.0, 0.0, 731.0, 1023.0], "bbox_mode": 0, "category_id": 0}, 
                {"bbox": [0.0, 0.0, 731.0, 1023.0], "bbox_mode": 0, "category_id": 0}]}
```

## Register Custom Datasets
Detectron2 comes up with trained models that can be downloaded in the [Detectron2 Model Zoo](https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md). The advantage is models are trained on large datasets that may already have objects from our datasets. This will save lot a time involved in learning from scratch only fine-tuning is required.

Models form ojective detection are trained on [COCO](https://cocodataset.org/#home) dataset, a large-scale object detection, segmentation, and captioning dataset with scenes containing common objects in their natural context.



## References:
Airnb  project article: https://medium.com/airbnb-engineering/amenity-detection-and-beyond-new-frontiers-of-computer-vision-at-airbnb-144a4441b72e

Detectron2: https://github.com/facebookresearch/detectron2

Data Download: https://www.learnopencv.com/fast-image-downloader-for-open-images-v4/


Experiment tracking: https://lambdalabs.com/blog/weights-and-bias-gpu-cpu-utilization/
