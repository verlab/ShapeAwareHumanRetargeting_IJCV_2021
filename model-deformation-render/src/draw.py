import numpy as np
import cv2
import pdb

def draw_texture_line (c_ori,c_end,l,A,input_image,output_image):
    if c_ori > c_end:
        tp = c_ori
        c_ori = c_end
        c_end = tp

    

    for x in range(int(c_ori),int(c_end + 0.5)+1):
        if x > 0 and x < output_image.shape[1]:
              my_point = (A.dot(np.array([l,x,1.0]).T)).astype(int)
              if my_point[0] > 0 and my_point[0] < input_image.shape[0] and my_point[1] > 0 and my_point[1] < input_image.shape[1]:
                  output_image[l,x,:] = input_image[my_point[0],my_point[1],:] 

def draw_triangle(tri,A,input_image,output_image):

    my_tri = tri.reshape((3,2))
    # sort points by y
    for i in range(3):
        v = np.copy(my_tri[i,:])
        id = i
        for j in range(i,3):
             if  my_tri[j,0] < v[0]:
                 v = np.copy(my_tri[j,:])
                 id  = j
        my_tri[id,:] = my_tri[i,:]
        my_tri[i,:] = v
    
    
    c_ori = my_tri[0,1]
    c_end = my_tri[0,1]
    
    # test colinear vertices

    if int(my_tri[0,0]) == int(my_tri[1,0]):
         c_end = my_tri[1,1]    
    
    # no problem if coef is inf
    with np.errstate(all='ignore'):
        d0 = (my_tri[1,1] - my_tri[0,1])/(my_tri[1,0] - my_tri[0,0])
        d1 = (my_tri[2,1] - my_tri[0,1])/(my_tri[2,0] - my_tri[0,0])
        d2 = (my_tri[2,1] - my_tri[1,1])/(my_tri[2,0] - my_tri[1,0])

    for l in range(0,int(my_tri[2,0] + 1 - my_tri[0,0] + 0.5)):
        my_line = int(my_tri[0,0] + l)  

        #if l == 14:
        #   pdb.set_trace()
   
        if my_line > 0 and my_line < output_image.shape[0]:
            #if abs(c_end - c_ori) > 100:
            #    pdb.set_trace() 
            draw_texture_line (c_ori,c_end,my_line,A,input_image,output_image)


        if (my_tri[0,0] + l + 1) < my_tri[1,0] :
            c_ori = c_ori + d1
            c_end = c_end + d0

        elif (my_tri[0,0] + l) < my_tri[1,0]:
            c_ori = c_ori + d1
            c_end = my_tri[1,1]
        elif (my_tri[0,0] + l + 1) < my_tri[2,0]: 
            c_ori = c_ori + d1
            c_end = c_end + d2 
        elif int(my_tri[2,0]) == int(my_tri[1,0]):
            c_ori = my_tri[2,1] 
            c_end = my_tri[1,1]

       
            
def warp_tri(tri1,tri2,input_image,output_image):
    A = cv2.getAffineTransform(tri2, tri1)
    #print A
    draw_triangle(tri2,A,input_image,output_image)
   
 
def test():

    input_image = cv2.imread("GOPR95340000000000.png")
    output_image = np.zeros_like(input_image)
    tri1 = np.float32([[540,540], [640,790], [990,940]])
    tri2 = np.float32([[540,540], [640,790], [990,940]])
    tri3 = np.float32([[400,200],[160,270], [400,400]])
    warpMat = cv2.getAffineTransform(tri2, tri1)
    warpMat2 = cv2.getAffineTransform(tri3, tri1)
    draw_triangle(tri2,warpMat,input_image,output_image)
    draw_triangle(tri3,warpMat2,input_image,output_image)
    cv2.imshow('ImageWindow', output_image)
    cv2.waitKey()

if __name__ == '__main__':
    test()

