import os
import cv2
import numpy as np
from numpy.core.numeric import zeros_like
from scipy.io.matlab.mio import mat_reader_factory
from scipy.io.matlab.mio5 import MatFile5Reader
from scipy.io.matlab.miobase import get_matfile_version
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from skimage.measure import label
from skimage.morphology import binary_closing


def remove_jp_paper(fragment_full_path, **kwargs):
    """removes japanese paper from fragment and save to new file
    
    Parameters
    ----------
    input_image : str
        The full path to the fragment image (.png file)
    kwargs : array tupple 
        settings struct (see param_segmentation.json)

    Returns
    -------
        The method writes the fragment without the japanese paper to a new image file 
        The path to the new image file is taken from settings struct kwargs "s_frag_no_jp_path"
    """
    img_name = fragment_full_path.split('/')[-1]
    plate_num=img_name.split('-')[0]   
    output_dir=kwargs.get('s_output_dir')
    
    if not os.path.exists(fragment_full_path):
        return

    Abgr = cv2.imread(fragment_full_path)
            
    
    A = cv2.cvtColor(Abgr, cv2.COLOR_BGR2RGB)
    Ag = cv2.cvtColor(A, cv2.COLOR_RGB2GRAY)
    maskA = Ag != 0
    lab_he = cv2.cvtColor(A, cv2.COLOR_RGB2LAB)
    Ahsv = cv2.cvtColor(A, cv2.COLOR_RGB2HSV)
    ab = Ahsv.astype(np.float64)

    Ifrag = Ag != 0
    ab_frag=ab[Ifrag]

    nColors = 5
    #repeat the clustering 3 times to avoid local minima
    kmeans = KMeans(n_clusters=nColors, random_state=0).fit(ab_frag)
    cluster_idx = kmeans.labels_
    cluster_center = kmeans.cluster_centers_
    
    pixel_labels = np.zeros(ab.shape[:2])
    pixel_labels[Ifrag] = cluster_idx + 1
    pixel_labels_frag = pixel_labels[Ifrag]
    #pixel_labels = np.reshape(pixel_labels,(nrows,ncols))

    #compute mean luminance of clusters
    #L=reshape(lab_he(:,:,1),nrows*ncols,1);
    L = lab_he
    In = {}
    for c_index in range(1, nColors + 1):
        In[c_index] = np.where(pixel_labels==c_index)
    #In[1] = np.where(pixel_labels==1)
    #In[2] = np.where(pixel_labels==2)
    #In[3] = np.where(pixel_labels==3)
    #I=np.stack([I1, I2, I3])
    jpIndex = findJpBySVMmodel(ab_frag,pixel_labels_frag,**kwargs)
    if len(jpIndex)==0:
        return

    pixel_labels1=np.zeros_like(pixel_labels)
    pixel_labels1[Ifrag] = 1
    for kk in range(len(jpIndex)):
        pixel_labels1[In[jpIndex[kk]]]=0

    largestCC = getLargestCC(pixel_labels1)
    filled = binary_closing(largestCC)
    
    output_image = Abgr * filled[..., np.newaxis]
    output_image = np.concatenate((output_image, filled[..., np.newaxis].astype(np.uint8) * 255), axis=-1)

    full_output_dir=os.path.join(output_dir, kwargs.get('s_frag_no_jp_path'),plate_num)
    if not os.path.exists(full_output_dir):
        os.makedirs(full_output_dir)

    #cv2.imwrite(os.path.join(output_dir, kwargs.get('s_frag_no_jp_path'),plate_num,fragment_full_path.split('/')[-1]), output_image)
    output_nojp_filename=os.path.join(full_output_dir,img_name)
    cv2.imwrite(output_nojp_filename, output_image)

    # _, axes = plt.subplots(1, 5, sharex=True, sharey=True)
    # axes[0].imshow(A)
    # axes[1].imshow(pixel_labels)
    # axes[2].imshow(Ahsv)
    # axes[3].imshow(pixel_labels1)
    # axes[4].imshow(largestCC)
    # plt.show()

    print('remove_jp_paper done')    


def getLargestCC(segmentation):
    labels = label(segmentation)
    assert( labels.max() != 0 ) # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC


def findJpBySVMmodel(ab_frag,pixel_labels_frag,**kwargs):
    jpIndex=[]
    # mat_file = '/home/itai/work/Adiel/JP_SVM_MODEL.mat'
    # mat_vars = scipy.io.loadmat(mat_file)
    mdl_beta = np.array([0.200024325764479, -0.202698672392475, 0.027450980391938])
    mdl_bias = 0.004912055360112
    score = ab_frag.dot(mdl_beta) + mdl_bias   
    labelIdx = (score > 0).astype(np.float32)
    all_Clusters = np.unique(pixel_labels_frag)
    mean_scores = []
    for cluster in all_Clusters:
        mean_scores.append(np.mean(labelIdx[pixel_labels_frag == cluster]))
    meanScores = np.stack(mean_scores)

    # in svm model 0=parchment and letters 1=japanese paper
    jp_svm_threshold=kwargs.get('s_jp_svm_threshold')
    jpIndex, =np.where(meanScores>=jp_svm_threshold)
   # print('MeanScore {} {} {},JP index={}'.format(meanScores[0],meanScores[1],meanScores[2],jpIndex))
    if len(jpIndex) == 0:
        print('NO Japanese Paper detected')
        return jpIndex
    return jpIndex + 1




def main():
    input_image = '/home/itai/work/Adiel/refactor/output/DSS_Fragments/fragments/417/417-Fg003-R-C01-R01-D29042013-T152419-LR445_ColorCalData_IAA_Both_CC110304_110702.png'
    output_dir = '/home/itai/work/Adiel/refactor/output'

    remove_jp_paper(input_image, output_dir)

if __name__ == '__main__':
    main()
