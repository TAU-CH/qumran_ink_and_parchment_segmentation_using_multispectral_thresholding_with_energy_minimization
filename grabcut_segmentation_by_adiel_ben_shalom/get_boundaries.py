import cv2
import os
from skimage.segmentation import find_boundaries
import matplotlib.pyplot as plt

def get_boundaries(img_base_path, img_name, **kwargs):
   # image_file = '/home/itai/work/Adiel/refactor/output/DSS_Fragments/fragments_no_jp/417/417-Fg003-R-C01-R01-D29042013-T152419-LR445_ColorCalData_IAA_Both_CC110304_110702.png'
    image_file=os.path.join(kwargs.get('s_output_dir'), img_base_path,img_name)
    image = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
    image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_bw = image_bw > 0
    boundries = find_boundaries(image_bw)
    _, axes = plt.subplots(1, 2, sharex=True, sharey=True)
    axes[0].imshow(image_bw, cmap='gray')
    axes[1].imshow(boundries)
    plt.show()