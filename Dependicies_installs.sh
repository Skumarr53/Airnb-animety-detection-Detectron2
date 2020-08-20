#!/bin/sh
pip install -U torch==1.4+cu100 torchvision==0.5+cu100 -f https://download.pytorch.org/whl/torch_stable.html 
pip install cython pyyaml==5.1
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

pip install awscli


pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu100/index.html

wget https://storage.googleapis.com/openimages/2018_04/class-descriptions-boxable.csv
2
 
3
wget https://storage.googleapis.com/openimages/2018_04/train/train-annotations-bbox.csv
4
 
5
wget https://storage.googleapis.com/openimages/2018_04/validation/validation-annotations-bbox.csv
6
 
7
wget https://storage.googleapis.com/openimages/2018_04/test/test-annotations-bbox.csv

# Download the trained model
!wget https://storage.googleapis.com/airbnb-amenity-detection-storage/airbnb-amenity-detection/open-images-data/retinanet_model_final/retinanet_model_final.pth 

# Download the train model config (instructions on how the model was built)
!wget https://storage.googleapis.com/airbnb-amenity-detection-storage/airbnb-amenity-detection/open-images-data/retinanet_model_final/retinanet_model_final_config.yaml

# Downloaded
Bathtub,Coffeemaker,Fireplace,Toilet,Swimming_pool,Bed,Billiard_table,Sink,Fountain,Oven,Ceiling_fan,Television,Microwave_oven,Gas_stove, Refrigerator,Kitchen_&_dining_room_table,Washing_machine,Stairs,Pillow,Mirror,Shower,Couch,Countertop,Dishwasher,Sofa_bed

# Downloading
Tree_house,Towel,Porch

#Yet to Download
,Wine_rack,Jacuzzi




3A15980218850513#5A593232343256335643004D6F746F2047200000#446F311B90758B67EA133CEAC2B35C0B9DABED1A#8CD98A16000000000000000000000000
