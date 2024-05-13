## Neural Network Analysis

The original (broken) CNN network can be found at original.py  
Fixed CNN network is at task2.py  

Fixes:  
CNN network consists from the feature extractor made of CNN layers, followed by the classifier made of fully connected layers.  
In the feature extractor, it is common practice that the number of channels grows (3->6->16) while spacial dimensions are reduced by pooling.  
The linkage between the feature extractor and the classifier is done via flattening activation map into a vector. In this network, the vector has 16x5x5=400 components, which assumes input images are 32x32x3. Intermediate size calculations are in code comments.  
