## Perception module

#### Traffic light detection
This module uses SSD (Single Shot MultiBox Detector) to detect traffic lights in the road images. We test with below 2 models (trained on COCO dataset and provided by Google's tensorflow team)
* [ssd_mobilenet_v1_coco](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17.tar.gz)
* [ssd_inception_v2_coco](http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2017_11_17.tar.gz)

Try running the inference code with one of the following commands
```
cd scripts

# Test on some sample images from simulator
python inference.py --image_dir ../data/sample/sim --viz

# Test on some sample images from site
python inference.py --image_dir ../data/sample/site --viz

# Average time taken by ssd_inception_v2 model (no visualization)
python inference.py --image_dir ../data/sample/sim --model ssd_inception_v2_coco_2017_11_17

# Average time taken by ssd_mobilenet_v1 model (no visualization)
python inference.py --image_dir ../data/sample/sim --model ssd_mobilenet_v1_coco_2017_11_17
```

**Examples on simulation data**
![Alt text](data/sample/sim_1.png?raw=true "Simulation Example")
![Alt text](data/sample/sim_2.png?raw=true "Simulation Example")

**Examples on site data**
![Alt text](data/sample/site_1.png?raw=true "Site Example")
![Alt text](data/sample/site_2.png?raw=true "Site Example")

#### Traffic light classification (simulator)

This module assumes that we have the crop for traffic light (obtained from localization module) and we have to classify it in one of the following categories `['UNKNOWN', 'GREEN', 'YELLOW', 'RED']`.

Training data for this exercise is obtained from one of the pinned items in the slack channel `#s-t3-p-system-integra` (shared by See on November 21, 2017). We store a local copy of the data at `$(CarND-Capstone)/ros/src/tl_detector/light_classification/data`

Data statistics (number in braces corresponds to the label to integer mapping as mentioned in `styx_msgs/msg/TrafficLight.msg`)
* UNKNOWN(4) - 6 images
* GREEN(2) - 35 images
* YELLOW(1) - 35 images
* RED(0) - 337 images

