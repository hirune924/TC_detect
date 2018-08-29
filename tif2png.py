#!/usr/bin/python 
# -*- coding: UTF-8 -*-


import glob
import subprocess
import os

def main():
    #TC = glob.glob('01_extract/train/nonTC/*')
    TC = glob.glob('01_extract/test_1/test/*')
    print(TC[:10])
    
    for idx, tif in enumerate(TC):
        tif_name = os.path.basename(tif)
        dir_name = os.path.dirname(tif)
        img_name = os.path.splitext(tif_name)[0]
        cvt_cmd = 'convert ' + tif + ' 02_png/test_1/test/' + img_name + '.png'
        if idx%1000 == 0:
            print(idx/float(len(TC)))
        subprocess.check_call(cvt_cmd, shell=True)

if __name__ == '__main__':
    main()
