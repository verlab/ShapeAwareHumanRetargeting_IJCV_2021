import numpy as np
import cv2


def visibility_match(image_m,images_motion_tex,images_source_tex):
    print image_m
    image = cv2.imread(image_m)
    ind = [0,0,0,0,0,0,0,0]
    dist0 = [10000000,10000000,10000000,10000000,10000000,10000000,10000000,10000000]
    #tex_original = cv2.imread(images_motion_tex[index],0)
    tex_original = cv2.imread(images_motion_tex,0)    
    ret,tex_original = cv2.threshold(tex_original,0,255,cv2.THRESH_BINARY)
    print "Doing distance"
    #print images_source_tex    
   
    for i,tex_source in enumerate(images_source_tex):
        dist = 0
        tex_now = cv2.imread(tex_source,0)
        ret,tex_now = cv2.threshold(tex_now,0,255,cv2.THRESH_BINARY)
        tex_now = cv2.bitwise_xor(tex_original,tex_now); 

        tex_now0 = tex_now[:int(tex_now.shape[0]*0.2),:tex_now.shape[1]/4]
        dist = cv2.countNonZero(tex_now0)
                                
        #print dist,dist0[0]
        if dist < dist0[0]:
            dist0[0] = dist
            ind[0] = i 
               
        dist = 0
        tex_now0 = tex_now[int(tex_now.shape[0]*0.2):tex_now.shape[0]/2,:tex_now.shape[1]/4]
        dist = cv2.countNonZero(tex_now0)
                                
               #print dist,dist0[0]
        if dist < dist0[1]:
            dist0[1] = dist
            ind[1] = i  
              
        dist = 0
        tex_now0 = tex_now[:int(tex_now.shape[0]*0.3),tex_now.shape[1]/4:tex_now.shape[1]/2]
        dist = cv2.countNonZero(tex_now0)
                               
        if dist < dist0[2]:
            dist0[2] = dist
            ind[2] = i
                 
        dist = 0

        tex_now0 = tex_now[int(tex_now.shape[0]*0.3):tex_now.shape[0]/2,tex_now.shape[1]/4:tex_now.shape[1]/2]
        dist = cv2.countNonZero(tex_now0)
                               
        if dist < dist0[3]:
            dist0[3] = dist
            ind[3] = i
                 
        dist = 0

        tex_now0 = tex_now[:tex_now.shape[0]/2,tex_now.shape[1]/2:3*tex_now.shape[1]/4]
        dist = cv2.countNonZero(tex_now0)
                            
        if dist < dist0[4]:
            dist0[4] = dist
            ind[4] = i
                 
        dist = 0
        tex_now0 = tex_now[:tex_now.shape[0]/2,3*tex_now.shape[1]/4:tex_now.shape[1]]
        dist = cv2.countNonZero(tex_now0)
                            
        if dist < dist0[5]:
            dist0[5] = dist
            ind[5] = i
                 
        dist = 0
        tex_now0 = tex_now[tex_now.shape[0]/2:tex_now.shape[0],:tex_now.shape[1]/2]
        dist = cv2.countNonZero(tex_now0)
                             
        if dist < dist0[6]:
            dist0[6] = dist
            ind[6] = i                 
        
        dist = 0
        tex_now0 = tex_now[tex_now.shape[0]/2:tex_now.shape[0],tex_now.shape[1]/2:]
        dist = cv2.countNonZero(tex_now0)
                             
        if dist < dist0[7]:
            dist0[7] = dist
            ind[7] = i

    return ind
