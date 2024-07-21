import os
import json
from remove_jp_paper import remove_jp_paper
from cut_fragment import cut_fragment
from get_boundaries import get_boundaries
from  save_boundaries_to_json import display_json, save_boundaries_to_json


RUNFROMTEXTFILE=1
# in this list files are in format:
list_to_process='test_images.txt'

def runner(image_full_path, **kwargs):
    run_cut_fragment = kwargs.get('s_run_cut_fragment')
    run_remove_jp_paper = kwargs.get('s_run_remove_jp_paper')
    run_write_boundaries = kwargs.get('s_run_write_boundaries')

    image_name=image_full_path.split('/')[-1]
    plate_num=image_name.split('-')[0]

    fragment_full_path=os.path.join(kwargs.get('s_output_dir'), kwargs.get('s_frag_path'),plate_num,image_name[0:-4]+'.png')
      
    if (run_cut_fragment):
        cut_fragment(image_full_path, **kwargs)
    
       
        # write fragment boundaries to json file  
        if (run_write_boundaries):
            json_path=kwargs.get('s_json_path')
            output_json_file=os.path.join(kwargs.get('s_output_dir'),json_path,plate_num,image_name[0:-4]+'.json')
            if os.path.exists(fragment_full_path):
                save_boundaries_to_json(fragment_full_path, output_json_file)


    if (run_remove_jp_paper):
        #output_image_path = os.path.join(frag_path,plate_num,outimgname)
        remove_jp_paper(fragment_full_path, **kwargs)

        if (run_write_boundaries):
            json_path=kwargs.get('s_json_npjp_path')
            output_json_file=os.path.join(kwargs.get('s_output_dir'),json_path,plate_num,image_name[0:-4]+'.json')
            fragment_no_jp_full_path=os.path.join(kwargs.get('s_output_dir'), kwargs.get('s_frag_no_jp_path'),plate_num,image_name[0:-4]+'.png')
            if os.path.exists(fragment_no_jp_full_path):
                save_boundaries_to_json(fragment_no_jp_full_path, output_json_file)
            if kwargs.get('s_show_boundaries'):
                display_json(output_json_file,fragment_no_jp_full_path)
    return

def main():

    json_file = os.path.join(os.path.dirname(__file__), 'param_segmentation.json')
    with open(json_file, mode='r') as fh:
        json_params = json.load(fh)

    #function_with_params(5, **json_params)

    if RUNFROMTEXTFILE:
        text_file = open(list_to_process , 'r')
        onlycolorimgs = text_file.readlines()

        img_path=json_params['s_img_path']


        for j in range(0,len(onlycolorimgs)):
            if onlycolorimgs[j]=='\n':
                continue
            
            img_parent_path = onlycolorimgs[j].split('/')[0]
            #print('img_parent_path='+img_parent_path)

            img_name=onlycolorimgs[j].split('/')[1]
            plate_num=img_name.split('-')[0]
            img_name=img_name[0:-1]  #removes end of line
     #       imgcolorname = os.path.join(img_path,plate_num,img_name)
            image_name = os.path.join(img_path,img_parent_path, img_name)
            print(j, 'Process image ', image_name)
          
            runner(image_name, **json_params)
            
       # img_name='417-Fg003-R-C01-R01-D29042013-T152419-LR445_ColorCalData_IAA_Both_CC110304_110702.tif'
       # plate_num=img_name.split('-')[0]
       # img_base_path=json_params['s_frag_no_jp_path']
       # img_name=os.path.join(plate_num,img_name[0:-4]+'.png')
       # get_boundaries(img_base_path, img_name, **json_params)
    
    #onlycolorimgs = '417/417-Fg003-R-C01-R01-D29042013-T152419-LR445_ColorCalData_IAA_Both_CC110304_110702.tif'
    #img_path = '/Users/adiel/Dropbox/Projects/TAU/DeadSeaScrolls/Segmentation/DSS_TestImages/DSS_IMAGES/'
    #image_name = os.path.join(img_path, onlycolorimgs)
    #output_dir = '/Users/adiel/Dropbox/Projects/TAU/DeadSeaScrolls/Segmentation/refactor/output'
    #cut_fragment(image_name, output_dir, debug=True)


if __name__ == '__main__':
    main()
