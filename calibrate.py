import cv2 
import numpy as np 

img_path = 'board.jpg'
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

def zoom_crop(img, zoom_factor):
    h, w, l = img.shape

    return img[int(1/4*zoom_factor*h): int(h - 1/4*zoom_factor*h), int(1/4*zoom_factor*w): int(w - 1/4*zoom_factor*w)]
 
def calibrate(image, checkboard_size, square_size):
    # chessboard dimensions
    nX = checkboard_size[0]
    nY = checkboard_size[1]
    # square_size = 0.0215 #m
    # checkboard_size = (nX,nY)

    # point containers
    object_points_3D = np.zeros((nX * nY, 3), np.float32)                                                  
    object_points_3D[:,:2] = np.mgrid[0:nX, 0:nY].T.reshape(-1, 2)
    object_points_3D = object_points_3D * square_size

    object_points = []
    image_points = []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

    # find pattern
    success, corners = cv2.findChessboardCorners(gray, checkboard_size, None)

    # draw pattern
    if success:
        cv2.drawChessboardCorners(image, checkboard_size, corners, success)

    # cv2.imshow("Image", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    object_points.append(object_points_3D)
    image_points.append(cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria))

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, gray.shape[::-1], None, None)

    return mtx, dist

image = cv2.imread(img_path)
mtx, dist = calibrate(image, (9, 6), 0.008)

undistorted_frame = cv2.undistort(image, mtx, dist)

print(dist)
print(mtx)

# cv2.imshow("test", zoom_crop(undistorted_frame, 0.7))
# cv2.waitKey(0)
# cv2.destroyAllWindows()