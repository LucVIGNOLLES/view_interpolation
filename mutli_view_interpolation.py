import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from calibrate import calibrate, zoom_crop

## Global ======

MAX_DISPARITY = 32
WINDOW_SIZE = 4
FILTER_THRESHOLD = 0.5

MORPH_FACTOR = 0.5

FLANN_INDEX_KDTREE = 1

IMGS_PATHS = ["f4.jpg", "f1.jpg"]
SHOW_IMAGES = True
VID_PATH = r"C:\Users\vigno\Pictures\Camera Roll\lil_vid.mp4"

INTERP_STEP = 0.1

DIST = np.array([[2.22506912e-01 -4.30101865e+00, 3.98415539e-03, 3.75766916e-04, 1.53179780e+01]])
CAM_MTX = np.array([[1.40815886e+03, 0.00000000e+00, 6.54448498e+02],
                    [0.00000000e+00, 1.40803980e+03, 3.48015657e+02],
                    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

## Functions ======

def distance(point, line): 
    a,b,c = line[0], line[1], line[2] 
    x,y = point[0], point[1] 
    return np.abs(a*x+b*y+c)/np.sqrt(a**2+b**2)

def distance_rato_filter_all_matches(matches, thresh):
    filtered_matches = [] 
 
    for match in matches:
        m = match[0]
        n = match[1]
        if m.distance < thresh * n.distance: 
            filtered_matches.append(m) 

    return filtered_matches

def make_homogeneous(vector):
    new_vec = np.zeros((len(vector), 3))
    for i in range(len(vector)):
        new_vec[i, 0] = vector[i,0]
        new_vec[i, 1] = vector[i,1]
        new_vec[i, 2] = 1

    return new_vec

def Ferror(F,pts1,pts2):   
    check = [np.dot(np.transpose(np.append(pts2[i], 1)), np.dot(F, np.append(pts1[i], 1))[0]) for i in range(len(pts1))]
    return np.abs(np.mean(check))

def interpolate(i, imgL, imgR, disparity):
    """
    :param i:
    :param imgL:
    :param imgR:
    :param disparity:
    :return:
    """
    ir = np.zeros_like(imgL)
    for y in range(imgL.shape[0]):
        for x1 in range(imgL.shape[1]):
            x2 = int(x1 - disparity[y, x1])
            x_i = int((2 - i) * x1 + (i - 1) * x2)
            if 0 <= x_i < ir.shape[1] and 0 <= x2 < imgR.shape[1]:
                ir[y, x_i] = int((2 - i) * imgL[y, x1] + (i - 1) * imgR[y, x2])
    return ir.astype(np.uint8)

def interpolateAndMask(alpha, imgl, imgr, disp):
    """
    Tuning alpha between 0 and 1 will create an
    interpolated view between in between imgl and imgr 
    """
    height, width, depth = imgl.shape
    interpolated = np.zeros_like(imgl) # Container for the new image
    mask = np.ones((height, width), dtype=np.uint8) # Container for the mask
    # For each pixel
    for y in range(height):
        for xl in range(width):
            xr = int(xl - disp[y, xl]) # Matching pixels according to the disparity map
            xInterp = int((1 - alpha) * xl + alpha * xr) # Alpha interpolate bothe pixels
            if 0 <= xInterp < interpolated.shape[1] and 0 <= xr < imgr.shape[1]:
                # Blend both pixels 
                for c in range(depth):
                    interpolated[y, xInterp, c] = int((1 - alpha) * imgl[y, xl, c] + alpha * imgr[y, xr, c])
                mask[y, xInterp] = 0 # Pixel isn't part of a hole
    return interpolated, mask

def interpolate_views(img1, img2):
    imsize = (img1.shape[1], img1.shape[0])
    # SIFT detector init
    sift = cv.SIFT_create()

    # FLANN tuning and init
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50) 
    flann = cv.FlannBasedMatcher(index_params,search_params)

    # BM matcher init
    disparity_matcher = cv.StereoSGBM_create(
        minDisparity=-MAX_DISPARITY,
        numDisparities=MAX_DISPARITY * 2,
        blockSize=5,
        P1=8 * 3 * WINDOW_SIZE ** 2,
        P2=32 * 3 * WINDOW_SIZE ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=0,
        speckleRange=2,
        preFilterCap=63,
        mode=cv.STEREO_SGBM_MODE_SGBM_3WAY
    )

    # Find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Match keypoints with FLANN
    matches = flann.knnMatch(des1, des2, k=2)

    best_matches = distance_rato_filter_all_matches(matches, FILTER_THRESHOLD)

    if SHOW_IMAGES:
        # Draw matches
        img_matches = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], 3), dtype=np.uint8)
        output = cv.drawMatches(img1, kp1, img2, kp2, best_matches, img_matches, flags=cv.DrawMatchesFlags_DEFAULT)

        # Detected good matches
        cv.imshow('img', output)
        cv.waitKey(0)
        cv.destroyAllWindows()

    print("Found", len(best_matches), "matches")

    # Select features that give best matches
    pts1 = np.int32([kp1[match.queryIdx].pt for match in best_matches])
    pts2 = np.int32([kp2[match.trainIdx].pt for match in best_matches])

    F, mask = cv.findFundamentalMat(pts1,pts2,cv.FM_LMEDS)
    print(F)

    print("F matrix error :", Ferror(F, pts1, pts2))

    # Select only inlier points
    pts1 = pts1[mask.ravel()==1]
    pts2 = pts2[mask.ravel()==1]

    # Compute homographies to rectify the images
    _, H1, H2 = cv.stereoRectifyUncalibrated(pts1,pts2, F, imsize)

    # Rectify images
    img1_rect = cv.warpPerspective(img1, H1, imsize, cv.INTER_LINEAR) 
    img2_rect = cv.warpPerspective(img2, H2, imsize, cv.INTER_LINEAR)
    h,w,_ = img1_rect.shape

    if SHOW_IMAGES:
        comp = np.concatenate((img1_rect, img2_rect), axis=1)
        cv.imshow("comp", comp)
        cv.waitKey(0)
        cv.destroyAllWindows()
    
    # Compute disparity map
    img1_rect_gray = cv.cvtColor(img1_rect, cv.COLOR_BGR2GRAY)
    img2_rect_gray = cv.cvtColor(img2_rect, cv.COLOR_BGR2GRAY)

    disparity = disparity_matcher.compute(img1_rect_gray, img2_rect_gray)

    norm_coeff = 255 / disparity.max() # Map needs to be normalized
    disparity = disparity * norm_coeff / 255

    if SHOW_IMAGES:
        plt.imshow(disparity)
        plt.show()

    interp_frames = []

    for alpha in np.arange(0, 1, INTERP_STEP):
        print ("\rProgress : ", int(100 *alpha), "%",)
        interpolated, mask = interpolateAndMask(alpha, img1_rect, img2_rect, disparity)
        interp_frames.append(cv.inpaint(interpolated, mask, 3, cv.INPAINT_TELEA))

    interp_frames.append(img2_rect)
    return interp_frames, (w,h)
    
def main(imgs):
    video_frames = []

    for i in range(len(imgs) -1):
        print("processing images ", i+1, " out of ", len(imgs)-1)
        frames, sz = interpolate_views(imgs[i], imgs[i+1])
        [video_frames.append(frame) for frame in frames]

    fourcc = cv.VideoWriter_fourcc(*'mp4v') 
    video = cv.VideoWriter('out.avi', fourcc, 10, sz)

    for i in range(10):
        [video.write(frame) for frame in video_frames]
        video_frames.reverse()

    [video.write(frame) for frame in video_frames]
    video.release()

    return 0

if __name__ == "__main__":
    imgs = [cv.imread(file_name) for file_name in IMGS_PATHS]

    image = cv.imread("board.jpg")
    mtx, dist = calibrate(image, (9, 6), 0.021)

    undist_imgs = []

    for img in imgs:
        undist_imgs.append(zoom_crop(cv.undistort(img, mtx, dist), 0.2))

    
    main(imgs)