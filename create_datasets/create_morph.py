import numpy as np
import cv2
import argparse
from tqdm import tqdm
import os
import sys
import dlib

DATASET_PATH = '/home/user/Datasets/face'

def warp_im(im, M, dshape):
    output_im = np.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output_im

def transformation_from_points(points1, points2):
    """
    Return an affine transformation [s * R | T] such that:
        sum ||s*R*p1,i + T - p2,i||^2
    is minimized.
    """
    # Solve the procrustes problem by subtracting centroids, scaling by the
    # standard deviation, and then using the SVD to calculate the rotation. See
    # the following for more details:
    #   https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem

    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)

    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = np.linalg.svd(points1.T * points2)

    # The R we seek is in fact the transpose of the one given by U * Vt. This
    # is because the above formulation assumes the matrix goes on the right
    # (with row vectors) where as our solution requires the matrix to be on the
    # left (with column vectors).
    R = (U * Vt).T

    return np.vstack([np.hstack(((s2 / s1) * R,
                                       c2.T - (s2 / s1) * R * c1.T)),
                         np.matrix([0., 0., 1.])])

def get_landmarks(im,detector,predictor):
    rects = detector(im, 1)

    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])


def get_args():
    parser = argparse.ArgumentParser(description="This script cleans-up noisy labels "
                                                 "and creates database for training.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output", "-o", type=str,
                        help="path to output database mat file")
    parser.add_argument("--img_size", type=int, default=64,
                        help="output image size")
    
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    output_path = args.output
    img_size = args.img_size

    # morph2 folder path
    mypath = os.path.join(DATASET_PATH, 'MORPH_nonCommercial')
    isPlot = False
    
    ids_list = []
    ages_list = []
    img_paths_list = []

    # file format
    # 'id_num', 'picture_num', 'dob', 'doa', 'race', 'gender', 'facial_hair', 'age', 'age_diff', 'glasses', 'photo\n'
    morph_csvfile = os.path.join(mypath,'morph_2008_nonCommercial.csv')
    with open(morph_csvfile) as f:
        for i, line in enumerate(f):
            # skip the header
            if i == 0:
                continue
            line = line.split(',') # list []
            id_num = line[0]
            age = int(line[7])
            img_path = line[10].replace('\n','')

            ids_list.append(id_num) # string
            ages_list.append(age)
            img_paths_list.append(img_path)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('./create_datasets/landmarks/shape_predictor_68_face_landmarks.dat')
    
    ref_img = cv2.imread(mypath+'/Album2/009055_1M54.JPG')
    landmark_ref = get_landmarks(ref_img,detector,predictor)
    
    FACE_POINTS = list(range(17, 68))
    MOUTH_POINTS = list(range(48, 61))
    RIGHT_BROW_POINTS = list(range(17, 22))
    LEFT_BROW_POINTS = list(range(22, 27))
    RIGHT_EYE_POINTS = list(range(36, 42))
    LEFT_EYE_POINTS = list(range(42, 48))
    NOSE_POINTS = list(range(27, 35))
    JAW_POINTS = list(range(0, 17))

    # Points used to line up the images.
    ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
                                   RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)


    out_ids = []
    out_ages = []
    out_img_paths = []
    out_imgs = []

    for list_idx in tqdm(range(len(img_paths_list))):
        img_name = img_paths_list[list_idx]
        
        input_img = cv2.imread(mypath+'/'+img_name)
        img_h, img_w, _ = np.shape(input_img)

        detected = detector(input_img,1)
        if len(detected) == 1:

            #---------------------------------------------------------------------------------------------
            # Face align

            landmark = get_landmarks(input_img,detector,predictor)
            M = transformation_from_points(landmark_ref[ALIGN_POINTS], landmark[ALIGN_POINTS])
            input_img = warp_im(input_img, M, ref_img.shape)

            #---------------------------------------------------------------------------------------------

            detected = detector(input_img, 1)
            if len(detected) == 1:
                faces = np.empty((len(detected), img_size, img_size, 3))
                for i, d in enumerate(detected):
                    x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
                    xw1 = max(int(x1 - 0.4 * w), 0)
                    yw1 = max(int(y1 - 0.4 * h), 0)
                    xw2 = min(int(x2 + 0.4 * w), img_w - 1)
                    yw2 = min(int(y2 + 0.4 * h), img_h - 1)
                    faces[i,:,:,:] = cv2.resize(input_img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))
                
                    if isPlot:
                        cv2.rectangle(input_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.rectangle(input_img, (xw1, yw1), (xw2, yw2), (0, 255, 0), 2)
                        img_clip = ImageClip(input_img)
                        img_clip.show()
                        key = cv2.waitKey(1000)
                    
                #only add to the list when faces is detected
                bgr_face = faces[0,:,:,:].astype('uint8')
                rgb_face = cv2.cvtColor(bgr_face, cv2.COLOR_BGR2RGB)
                
                out_ids.append(ids_list[list_idx])
                out_ages.append(ages_list[list_idx])
                out_img_paths.append(img_paths_list[list_idx])
                out_imgs.append(rgb_face)

    np.savez(output_path,
             id=np.array(out_ids),
             image=np.array(out_imgs),
             age=np.array(out_ages),
             img_path = np.array(out_img_paths),
             img_size=img_size)

if __name__ == '__main__':
    main()