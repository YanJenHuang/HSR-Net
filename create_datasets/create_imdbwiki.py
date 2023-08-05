import numpy as np
import os
import cv2
import scipy.io
import argparse
from tqdm import tqdm
from TYY_utils import get_meta

DATASET_PATH = '/home/user/Datasets/face'

def get_args():
    parser = argparse.ArgumentParser(description="This script cleans-up noisy labels "
                                                 "and creates database for training.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="path to output database mat file")
    parser.add_argument("--db", type=str, default="wiki",
                        help="dataset; wiki or imdb")
    parser.add_argument("--img_size", type=int, default=64,
                        help="output image size")
    parser.add_argument("--min_score", type=float, default=1.0,
                        help="minimum face_score")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    output_path = args.output
    db = args.db
    img_size = args.img_size
    min_score = args.min_score

    # imdb or wiki folder path
    root_path = os.path.join(DATASET_PATH, "{}_crop/".format(db))
    mat_path = root_path + "{}.mat".format(db)
    if db == 'imdb':
        full_path, id_num, dob, gender, photo_taken, face_score, second_face_score, age = get_meta(mat_path, db)
    elif db == 'wiki':
        full_path, dob, gender, photo_taken, face_score, second_face_score, age = get_meta(mat_path, db)
    else:
        raise ValueError('db should be "imdb" or "wiki"')

    out_ids = []
    out_ages = []    
    out_img_paths = []
    out_imgs = []

    for i in tqdm(range(len(face_score))):

        if face_score[i] < min_score:
            continue

        if (~np.isnan(second_face_score[i])) and second_face_score[i] > 0.0:
            continue

        if ~(0 <= age[i] <= 100):
            continue

        if np.isnan(gender[i]):
            continue
        
        img_path = str(full_path[i][0])
        out_img_paths.append(img_path)
        if db == 'imdb':
            out_ids.append(id_num[i])
        else: # wiki
            out_ids.append(-1)

        bgr_face = cv2.imread(root_path + img_path)
        rgb_face = cv2.cvtColor(bgr_face, cv2.COLOR_BGR2RGB)
                    
        out_ages.append(age[i])
        out_imgs.append(cv2.resize(rgb_face, (img_size, img_size)))

    np.savez(output_path,
             id=np.array(out_ids),
             image=np.array(out_imgs),
             age=np.array(out_ages),
             img_path = np.array(out_img_paths),
             img_size=img_size)

if __name__ == '__main__':
    main()