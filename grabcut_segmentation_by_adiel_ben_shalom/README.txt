Fragment Segmentaion
====================

1. The DSS fragment segmentation process extracts the fragment from the TIF images that contains fragment, as
well as ruller, color ruler and fragment label.
2. The output of the process is json file with the coordinates of polygons
that comprises the boundaries of the fragment. 
Both the boundary of the fragment surface as well as holes (if exists) in the fragment
3. The process is comprisd of several methods

a. cut_fragment - takes as input the tif image and saves to temporary file
only the fragment. The methods also extracts the fragment from the background
using GrabCut algorithm. (https://docs.opencv.org/master/d8/d83/tutorial_py_grabcut.html)

b. remove_jp_paper - takes the fragment as input and removes the japanese rise paper that
is somethimes attached to the fragment. 
This methods uses color sperataion using trained SVM model to separate the fragment into 
parchment, ink and japanese paper
The method saves to a temporary directory the fragment without the 
japaneses paper

c. save_boundaries_to_json - traces the boundaries of the fragment and writes
them as polygons to a json file
The polygon of the fragment surface is written counterclock wise and polygons
of holes are written clockwise. 


4. runner.py contains an example how to run the process. 
In this example files to run are in file 'test_images.txt'


configuration file
==================
The segmentaiton processs uses configuration file param_segmentation.json

one only needs to change
s_img_path: location of TIFF DSS images
s_output_dir: location of output directory

settings to control which funtion to invoke: 
s_run_cut_fragment": 1 
s_run_remove_jp_paper": 1
s_run_write_boundaries": 1 - note that boundaries can be computed to both the original fragment and the fragment after japanese paper removal process
s_show_boundaries": 0,

    


