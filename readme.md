# Airbnb's Amenity Detection with Detectron2

- [Airbnb's Amenity Detection with Detectron2](#airbnbs-amenity-detection-with-detectron2)
  - [Overview](#overview)
  - [Dataset description](#dataset-description)
  - [References:](#references)

## Overview



## Dataset description

Google's public available Open Images V5 has been used which contains 15.4M annotated bounding boxes for over 600 object categories. It has 1.9M images and is largest among all existing datasets with object location annotations. The classes include a variety of objects in various categories. It covers classes varying from different kinds of musical instruments(e.g. organ, cello, piano etc.) to different kinds of aquatic animals(e.g. goldfish, crab, seahorse, oyster etc.) to various kinds of kitchenware(e.g. spoon, kitchen knife, frying pan, dishwasher) and so on.

Since our focus is on animitey detection filtered out images corresponding to objects that don't fall in this category. . Below are choosen subset classes, the ones which relate most to Airnb business.

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



## References:

https://www.learnopencv.com/fast-image-downloader-for-open-images-v4/

https://github.com/facebookresearch/detectron2 

wandb: https://lambdalabs.com/blog/weights-and-bias-gpu-cpu-utilization/
