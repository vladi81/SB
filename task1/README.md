## General  
To install detectron2 run:  
python3 -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

Run task1.py  

The input dataset is at ./data  
Images with person in the sea are copied to ./out_data_with  
Images without person in the sea are copied to ./out_data_without  
In addition, ./viz_data is created with vizualization of person detections, sea and sky segmentations for each input image.  
Log messages were saved at log.txt

## Details
People in images where detected using 'person' class.
The detection threshold of 0.3 was used, reducing false positives but sensitive enough to detect people in the sea.  
In some cases, the sea was partially or fully detected as water or river. To handle better such cases, sea/water/river classes were combined into a single mask.  
In some cases, the sea was split into several segments due to an object in the middle. To improve handling, 2nd largest sea segment was added to the analysis if it is comparable in size to the largest sea segment.  
In some cases there were strong reflections of the sky in the sea, so that the sea was assigned 'sky' class label. Heuristical approach was used to find such cases - sky in the bottom of an image.
