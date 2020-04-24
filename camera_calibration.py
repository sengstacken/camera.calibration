# file to develop pin hole camera model from test images using openCV
#
# Aaron Sengstacken - 3.31.2017

# import shiz
import numpy as np
import cv2
#import matplotlib.pyplot as plt
import glob

def camera_cal(directory_of_cal_images):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0).  Note object points are 3-D of the real world
    # space, assume that the chess board was kept stationary and the camera was moved.

    # chessboard size, count starting w/ zero
    num_sq_width = 9
    num_sq_height = 6

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


objpoints, imgpoints, chessboards, im_size = camera_cal('cal_images/')

# find the camera model
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints,imgpoints,im_size, None, None)

# save camera model
np.savez("ccal.npz",ret=ret,mtx=mtx,dist=dist,rvecs=rvecs,tvecs=tvecs,objpoints=objpoints,imgpoints=imgpoints,im_size=im_size)

# process an image
image_correct('cal_images/IMG_3181.jpg',mtx,dist)