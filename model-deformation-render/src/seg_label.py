import numpy as np
import cv2
import pdb

def flood_flag(flag,x,y,label,mask):

    my_list = [[x,y]]

    while len(my_list) > 0:
        #print my_list
        point = my_list[0]
        my_list.pop(0)
        #cv2.imshow('ImageWindow',mask)
        #cv2.waitKey()
        if point[0] < 0 or point[0]  >= label.shape[0] or point[1]  < 0 or point[1]  >= label.shape[1]:
            continue  
        if mask[point[0],point[1]] == 0:
            continue

        mask[point[0],point[1]] = 0

        label[point[0],point[1]] = flag

        my_list.append([point[0]-1,point[1]])
        my_list.append([point[0]+1,point[1]])
        my_list.append([point[0],point[1]-1])
        my_list.append([point[0],point[1]+1])
        



def flood_close_flag_1(flag,x,y,label,mask):

    my_list = [[x,y]]

    if flag != 45 and flag != 60 and flag != 75 and flag != 90:
        return mask,flag

    while len(my_list) > 0:
        #print my_list
        point = my_list[0]
        my_list.pop(0)
        #cv2.imshow('ImageWindow',mask)
        #cv2.waitKey()
        if point[0] < 0 or point[0]  >= label.shape[0] or point[1]  < 0 or point[1]  >= label.shape[1]:
            continue  
        if mask[point[0],point[1]] == 255:
            continue

        mask[point[0],point[1]] = 255
 
        if flag == 45:
            if label[point[0],point[1]] == 45 or label[point[0],point[1]] == 60:
               return mask,label[point[0],point[1]]

        if flag == 60:
            if label[point[0],point[1]] == 75 or label[point[0],point[1]] == 90 or label[point[0],point[1]] == 195 or label[point[0],point[1]] == 210:
               return mask,label[point[0],point[1]]

        if flag == 75:
            if label[point[0],point[1]] == 105 or label[point[0],point[1]] == 120:
               return mask,label[point[0],point[1]]
 
        if flag == 90:
            if label[point[0],point[1]] == 135 or label[point[0],point[1]] == 150 or label[point[0],point[1]] == 165 or label[point[0],point[1]] == 180:
               return mask,label[point[0],point[1]]

        my_list.append([point[0]-1,point[1]])
        my_list.append([point[0]+1,point[1]])
        my_list.append([point[0],point[1]-1])
        my_list.append([point[0],point[1]+1])
        
    return mask,0

def build_seg(my_segs):
    
    if len(my_segs) == 2:     
        return my_segs[0]

    image_label = my_segs[0]

    image_label_0 = np.zeros(my_segs[0].shape,dtype=np.uint8)

    for i in range(14):
        label = np.where(image_label == (i+1)*15,np.ones(image_label.shape,dtype=np.uint8)*255,np.zeros(image_label.shape,dtype=np.uint8))
        im2, contours, hierarchy = cv2.findContours(label, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)      
        area = 0.0
        id = -1
        for j in range(len(contours)):
            area2 = cv2.contourArea(contours[j])
            if area2 > area:
                id = j
                area = area2 
 
        #to_draw = np.zeros((label.shape[0],label.shape[1],3),dtype=np.uint8)       
        cv2.drawContours(image_label_0, contours[id:id+1], -1, (i+1)*15, -1)


    if len(my_segs) == 1:     
        return image_label_0

    image_label = np.where(my_segs[1] > 5,np.ones(image_label.shape,dtype=np.uint8)*255,np.zeros(image_label.shape,dtype=np.uint8)) 
    im2, contours, hierarchy = cv2.findContours(image_label, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  
    area = 0.0
    id = -1

    for j in range(len(contours)):
       area2 = cv2.contourArea(contours[j])
       if area2 > area:
           id = j
           area = area2 
    dilate_area = np.sqrt(area)    
     
    image_label_1 = np.zeros(my_segs[0].shape,dtype=np.uint8)
    cv2.drawContours(image_label_1, contours[id:id+1], -1, 255, -1)

    image_label = np.where(my_segs[2] > 5,np.ones(image_label.shape,dtype=np.uint8)*255,np.zeros(image_label.shape,dtype=np.uint8)) 
    im2, contours, hierarchy = cv2.findContours(image_label, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  
    area = 0.0
    id = -1

    for j in range(len(contours)):
       area2 = cv2.contourArea(contours[j])
       if area2 > area:
           id = j
           area = area2   
     
    image_label_2 = np.zeros(my_segs[0].shape,dtype=np.uint8)
    cv2.drawContours(image_label_2, contours[id:id+1], -1, 255, -1)

    #mask_0 = np.zeros_like(image_label_1)

    for i in range(image_label_1.shape[0]):
       for j in range(image_label_1.shape[1]):
          #print i,j
          if image_label_1[i,j] == 0:
             continue

          mask,flag = flood_close_flag_1(my_segs[1][i,j],i,j,image_label_0,np.zeros_like(image_label_1)) 
          image_label_1[i,j] = flag


    cv2.imshow('ImageWindow',image_label_1)
    cv2.waitKey()

  

    return my_segs[0]


 
def test():

    image_label_2 = cv2.imread("GOPR95340000000125_2.png",0)
    image_label_1 = cv2.imread("GOPR95340000000125_1.png",0)
    image_label_0 = cv2.imread("GOPR95340000000125_0.png",0)    
    labels = build_seg([image_label_0,image_label_1,image_label_2])


if __name__ == '__main__':
    test()

