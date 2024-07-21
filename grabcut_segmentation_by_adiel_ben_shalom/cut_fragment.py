import os
from remove_jp_paper import remove_jp_paper
import numpy as np
import cv2



def cut_fragment(input_image, **kwargs):
    """cut only the fragment from original DSS image

    Parameters
    ----------
    input_image : str
        The full path to image location (with image name)
    kwargs : array tupple 
        settings struct (see param_segmentation.json)

    Returns
    -------
        The method writes the cutted fragment to a new image file 
        The path to the new image file is taken from settings struct kwargs "s_frag_path"
    """

    DEBUG = kwargs.get('s_debug')
    binthresh=kwargs.get('s_binthresh')
    fill_holes=kwargs.get('s_fill_holes')
    output_dir=kwargs.get('s_output_dir')

    #binthresh=80
    #fill_holes=1

    base_path = output_dir
    frag_path  = os.path.join(base_path, kwargs.get('s_frag_path'))
    debug_path = os.path.join(base_path, kwargs.get('s_debug_path'))
    frag_cords = os.path.join(base_path, kwargs.get('s_frag_cords'))
    frag_no_jp_path  = os.path.join(base_path, kwargs.get('s_frag_no_jp_path'))

    img_name = input_image.split('/')[-1]
    if DEBUG:
        print('input_image='+ input_image)

    plate_num=img_name.split('-')[0]   

    try: 
        imgorig = cv2.imread(input_image)
        h, w = imgorig.shape[:2]
        if DEBUG:
            print('image size='+ str(h)+ ' ' + str(w))

        resize_factor = kwargs.get('s_resize_factor')
        gc_bg_rect = kwargs.get('s_gc_bg_rect')  #40  # for resize_factor 0.5 I used 20 , for 1 use 40

        h = h * resize_factor
        w = w * resize_factor

        while (h*w>10000000):
            resize_factor = resize_factor*0.5
            gc_bg_rect = gc_bg_rect/2
            gc_bg_rect = gc_bg_rect.__int__()
            h = h * resize_factor
            w = w * resize_factor

        imgscaled = cv2.resize(imgorig, None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_CUBIC)
        if DEBUG:
            print('image size=' + str(h) + ' ' + str(w))

        imggray = cv2.cvtColor(imgscaled, cv2.COLOR_BGR2GRAY)
        if DEBUG:
            outimgname = img_name[0:-4] + '_gc_img_gray.png'
            if not os.path.exists(os.path.join(debug_path,plate_num)):        
                os.makedirs(os.path.join(debug_path,plate_num))
            cv2.imwrite(os.path.join(debug_path,plate_num,outimgname), imggray)

        blur = cv2.GaussianBlur(imggray, (5, 5), 0)
        ret3, imbinary = cv2.threshold(blur, binthresh, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(30,30))
        imbinary = cv2.morphologyEx(imbinary,cv2.MORPH_CLOSE,kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
        imbinary = cv2.morphologyEx(imbinary, cv2.MORPH_OPEN, kernel)

        if DEBUG:
            outimgname = img_name[0:-4] + '_gc_img_bin0.png'
            cv2.imwrite(os.path.join(debug_path, plate_num, outimgname), imbinary)
        im_floodfill = imbinary.copy()
        im_out = imbinary

        if DEBUG:
            outimgname = img_name[0:-4] + '_gc_img_bin.png'
            cv2.imwrite(os.path.join(debug_path, plate_num, outimgname), im_out)

        imbinary = im_out

        #---- Connected component
        # You need to choose 4 or 8 for connectivity type
        connectivity = 4
        # Perform the operation
        output = cv2.connectedComponentsWithStats(imbinary, connectivity, cv2.CV_32S)
        # Get the results
        # The first cell is the number of labels
        num_labels = output[0]
        # The second cell is the label matrix
        labels = output[1]
        # The third cell is the stat matrix
        stats = output[2]
        stats = stats[1:,] #remove background label
        # The fourth cell is the centroid matrix, x,y locations (column, row)
        centroids = output[3]
        centroids = centroids[1:,] #remove background label

        d=[]
        dict={}
        minCCArea=0.001*w*h
        imgcenter = np.divide(imgscaled.shape[0:2],2) #row,column
        if DEBUG:
            print('imgcenter=')
            print (imgcenter)
            print('minCCWidth=', minCCArea)
        for k in range(0, centroids.shape[0]):
            if DEBUG:
                print(k,'area=', stats[k,cv2.CC_STAT_AREA])
            ccArea=stats[k,cv2.CC_STAT_AREA]
            if ccArea > minCCArea:
                if DEBUG:
                    print (k, centroids[k])
                dist=np.linalg.norm(np.flip(imgcenter,0) - np.array(centroids[k]))
                if DEBUG:
                    print (dist)
                d.append(dist)
                dict[k]=dist

        if (len(dict)==0):
            print('ERROR dict size is 0 ')
            raise 'Error!'

        mn = min(dict.items(), key=lambda x: x[1])
        if DEBUG:
            print( 'mn=',mn)
        cc_ind=mn[0]
        if DEBUG:
            print ('cc_ind=' + str(cc_ind))
            print ('cc boundaries=')
            print (stats[cc_ind])

        # connected component mask - big image size
        ccmask = np.where((labels == cc_ind+1), 1, 0).astype('uint8')

        #check if connected component is all image. This means that there was an error that resulted in connected component that is entire image.
        h_, w_ = imbinary.shape
        if stats[cc_ind][0] == 0 or stats[cc_ind][2] == w_ or stats[cc_ind][3] == h_:
            print('ERROR CC is entire fragment ')
            raise 'Error!'
            

        # fill holes in mask
        lab_val=1
        #largest_obj_lab = np.argmax(stats[1:, 4]) + 1
        largest_mask = np.zeros(imbinary.shape, dtype=np.uint8)
        largest_mask[labels == cc_ind+1] = lab_val
        fill_holes=1
        if fill_holes:
            bkg_locs = np.where(labels == 0)
            bkg_seed = (bkg_locs[0][0], bkg_locs[1][0])
            img_floodfill = largest_mask.copy()
            h_, w_ = largest_mask.shape
            mask_ = np.zeros((h_ + 2, w_ + 2), dtype=np.uint8)
            cv2.floodFill(img_floodfill, mask_, seedPoint=bkg_seed, newVal=lab_val)
            holes_mask = cv2.bitwise_not(img_floodfill)  # mask of the holes.
            holes_mask1 = np.where((holes_mask == 255),1,0)
            ccmask = ccmask + holes_mask1


        if DEBUG:
            outimgname = img_name[0:-4] + '_gc_ccmask.png'
            cv2.imwrite(os.path.join(debug_path, plate_num, outimgname), ccmask*255)
            outimgname = img_name[0:-4] + '_gc_labels.png'
            cv2.imwrite(os.path.join(debug_path, plate_num, outimgname), labels*255)
            outimgname = img_name[0:-4] + '_gc_mask_fill_holes.png'
            cv2.imwrite(os.path.join(debug_path, plate_num, outimgname), ccmask * 255)


        if (stats[cc_ind][1]-gc_bg_rect < 0) or (stats[cc_ind][0]-gc_bg_rect<0):
            print ('ERROR while cropping ')
            raise 'Error!'

        crop_img=imgscaled[stats[cc_ind][1]-gc_bg_rect:stats[cc_ind][1]+stats[cc_ind][3]+gc_bg_rect*2,\
                            stats[cc_ind][0]-gc_bg_rect:stats[cc_ind][0]+stats[cc_ind][2]+gc_bg_rect*2]

        if DEBUG:
            outimgname = img_name[0:-4] + '_gc_img_cc.png'
            cv2.imwrite(os.path.join(debug_path, plate_num, outimgname), crop_img)

        # crop image mask
        ccmask_croped = ccmask[stats[cc_ind][1] - gc_bg_rect:stats[cc_ind][1] + stats[cc_ind][3] + gc_bg_rect * 2, \
                    stats[cc_ind][0] - gc_bg_rect:stats[cc_ind][0] + stats[cc_ind][2] + gc_bg_rect * 2]
        if DEBUG:
            outimgname = img_name[0:-4] + '_gc_ccmask_cropped.png'
            cv2.imwrite(os.path.join(debug_path, plate_num, outimgname), ccmask_croped*255)


        # write crop coordinates row, column
        outcordsname = img_name[0:-4] + '_gc_cords.txt'
        if not os.path.isdir(os.path.join(frag_cords,plate_num)):
            print('new directry has been created in cords '+ os.path.join(frag_cords,plate_num))
            os.makedirs(os.path.join(frag_cords,plate_num).replace(" ", "_"))
        cords_file = open(os.path.join(frag_cords,plate_num, outcordsname), 'w')
        cords_file.write(str(stats[cc_ind][1]-gc_bg_rect)+' '+str(stats[cc_ind][0]-gc_bg_rect)+' '+str(resize_factor))

        crop_mask = np.zeros(crop_img.shape[:2], np.uint8)
        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)

        cimg_w = crop_img.shape[1]
        cimg_h = crop_img.shape[0]

        if DEBUG:
            print ('run grabcut - rect')
            print ('crop_img.shape')
            print( crop_img.shape)
        rect = (gc_bg_rect , gc_bg_rect , cimg_w - gc_bg_rect , cimg_h - gc_bg_rect )
        if DEBUG:
            print (rect)

        cv2.grabCut(crop_img, crop_mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((crop_mask==2)|(crop_mask==0),0,1).astype('uint8')
        img = crop_img*mask2[:,:,np.newaxis]
        if DEBUG:
            outimgname = img_name[0:-4] + '_img_grabcut.png'
            cv2.imwrite(os.path.join(debug_path, plate_num, outimgname), img)
            outimgname = img_name[0:-4] + '_mask_grabcut.png'
            cv2.imwrite(os.path.join(debug_path, plate_num, outimgname), mask2*255)

        # apply connected components again
        # ---- Connected component
        # You need to choose 4 or 8 for connectivity type
        connectivity = 4
        # Perform the operation
        output = cv2.connectedComponentsWithStats(mask2, connectivity, cv2.CV_32S)
        # Get the results
        # The first cell is the number of labels
        num_labels = output[0]
        # The second cell is the label matrix
        labels = output[1]
        # The third cell is the stat matrix
        stats = output[2]
        stats = stats[1:, ]  # remove background label
        mxCCArea = 0
        for k in range(0, num_labels-1):
            #if DEBUG:
            #    print(k,'area1=', stats[k,cv2.CC_STAT_AREA])
            ccArea=stats[k,cv2.CC_STAT_AREA]
            if ccArea > mxCCArea:
                mxCCArea = ccArea
                mxAreaLabel = k+1
        mask3 = np.where((labels == mxAreaLabel), 1, 0).astype('uint8')
        if DEBUG:
            print('grabcut max area cc label=' + str(mxAreaLabel))
            print('grabcut max area =' + str(mxCCArea))
            outimgname = img_name[0:-4] + '_img_grabcut_biggest_cc.png'
            cv2.imwrite(os.path.join(debug_path, plate_num, outimgname), mask3*255)

        if DEBUG:
            print (img.shape)
            print (ccmask_croped.shape)

        img1=np.zeros(img.shape)
        img1[:,:,0] = img[:,:,0]*mask3
        img1[:, :, 1] = img[:, :, 1] * mask3
        img1[:, :, 2] = img[:, :, 2] * mask3
        img1 = np.concatenate((img1, 255 * mask3[..., np.newaxis]), axis=-1)
        outimgname = img_name[0:-4]  + '.png'

        if not os.path.isdir(os.path.join(frag_path,plate_num)):
            print('new directry has been created ' + os.path.join(frag_path,plate_num))
            os.makedirs(os.path.join(frag_path,plate_num).replace(" ", "_"))

        output_image_path = os.path.join(frag_path,plate_num,outimgname)
        cv2.imwrite(output_image_path,img1)
        if DEBUG:
            cv2.imwrite(os.path.join(debug_path, plate_num, outimgname), img1)

        print('cut_fragment done')    

        #remove_jp_paper(output_image_path, os.path.join(frag_no_jp_path, plate_num))
    except Exception as e:
        #logf.write('Error {0} {1} {2}\n'.format(str(j), str(imgcolorname), str(e)))
        #logf.flu
        print('**** ERROR ****')
    finally:
        pass


