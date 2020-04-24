# file to develop pin hole camera model from test images using openCV
#
# Aaron Sengstacken - 3.31.2017

# import shiz
import numpy as np
import cv2
#import matplotlib.pyplot as plt
import glob
import math

def camera_cal(directory_of_cal_images):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0).  Note object points are 3-D of the real world
    # space, assume that the chess board was kept stationary and the camera was moved.



    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((num_sq_height*num_sq_width,3), np.float32)
    objp[:,:2] = np.mgrid[0:num_sq_width, 0:num_sq_height].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.
    chessboards = [] # array of chessboard images

    # find all cal images
    images = glob.glob(directory_of_cal_images+'*.jpg')

    # loop over all cal images and search for test pattern (chessboard)
    for i,fname in enumerate(images):
        print(fname)
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # find corners
        ret,corners = cv2.findChessboardCorners(gray,(num_sq_width,num_sq_height),None)

        # If found, add object point, image points
        if ret == True:
            print('Found')
            objpoints.append(objp)

            # refine solution (note - This is optional, not required)
            cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(img,(num_sq_width,num_sq_height),corners,ret)
            chessboards.append(img)

            if i==0:
                #cv2.imshow('img',img)
                cv2.imwrite(fname[:-4]+'chess.jpg',img)
                #cv2.waitKey(500)
        else:
            print('Not Found')

    cv2.destroyAllWindows()
    return objpoints, imgpoints, chessboards, img.shape[0:2]


def image_correct(image_name,mtx,dist):
    img = cv2.imread(image_name)
    h,w = img.shape[:2]
    alpha = 1
    newcameramtx,roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),alpha,(w,h))
    dst = cv2.undistort(img,mtx,dist,None,newcameramtx)
    cv2.imwrite(image_name[:-4]+'cal.png',dst)


# chessboard size, count starting w/ zero
num_sq_width = 9
num_sq_height = 6
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((num_sq_height*num_sq_width,3), np.float32)
objp[:,:2] = np.mgrid[0:num_sq_width, 0:num_sq_height].T.reshape(-1,2)

# load camera model
ccal = np.load("ccal.npz")

# find all cal images
images = glob.glob('pos_images/'+'*.jpg')

# loop over all cal images and search for test pattern (chessboard)
for i,fname in enumerate(images):
    print(fname)

    # load image
    image = cv2.imread(fname)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    # find the chessboard in the image
    ret,corners = cv2.findChessboardCorners(gray,(num_sq_width,num_sq_height),None)

    if ret == True:
        print('Found chessboard')
        
        # refine solution (note - This is optional, not required)
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

        # find ratiation and translation vectors
        ret,rvecs,tvecs = cv2.solvePnP(objp, corners2, ccal['mtx'], ccal['dist'])

        print(ret)
        print("Rotation Vector:  ",rvecs)
        print("Translation Vector:  ",tvecs)

        # convert to 3x3 matrix
        rmat, jacobian = cv2.Rodrigues(rvecs)
        print("RMAT",rmat)
        print(rmat.shape)
        # convert from world coords to camera coords
        camera_pos = -np.matrix(rmat).T*np.matrix(tvecs)
        print(camera_pos)
        

        roll = math.atan2(-rmat[2][1],rmat[2][2])*180/math.pi
        pitch = math.asin(rmat[2][0])*180/math.pi
        yaw = math.atan2(-rmat[1][0],rmat[0][0])*180/math.pi
        print(roll,pitch,yaw)
        #projection matrix
        P = np.hstack((rmat,tvecs))
        euler_angles_degrees = -cv2.decomposeProjectionMatrix(P)[6]
        print("Pitch,Yaw,Roll",euler_angles_degrees)


        cv2.imshow('img',image)
        cv2.waitKey(500)

        #print(euler_angles_radians)
        #print(euler_angles_degrees)
    # save camera model
    #np.savez("ccal.npz",ret=ret,mtx=mtx,dist=dist,rvecs=rvecs,tvecs=tvecs,objpoints=objpoints,imgpoints=imgpoints,im_size=im_size)
