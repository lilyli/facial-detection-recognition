import numpy as np
import re
from skimage.transform import resize
import warnings
from skimage import img_as_ubyte
import imageio
import random
import pickle
from facial_detection_util import load_image


def get_imgs_bbxs(txt_file):
    ''' Given a text file, extract the image files and the bounding box coordinates'''
    '''Returns two lists: a list of the names of the image files and a list of the bounding boxes'''
    # open annotation file as list of strings
    with open(txt_file, 'r') as f:
        lines = f.read().splitlines()
    imgs = {} # dict mapping img files to bounding boxes of faces in the image

    for i in range(len(lines)):
        l = lines[i] 
        if l[-4:] == '.jpg':
            img = l
            num_bbx = int(lines[i + 1]) # next line
            bbxs = []
            for bbx_line in lines[i + 2:i + 2 + num_bbx]: # bbxs start 2 lines after img file
                bbx_coord = bbx_line.split(' ')
                bbx = bbx_coord[:4] # x1, y1, w, h
                if int(bbx[2]) >= 32 and int(bbx[3]) >= 32:
                # check width and height of bbx >= 32 to check face resolution
                    bbxs.append([int(n) for n in bbx])
            if len(bbxs) > 0:
                imgs[img] = bbxs
    
    return imgs



def generate_pos_neg_data(read_directory, imgs_bbxs, write_directory):
    x = 0
    for img_file, bbxs in imgs_bbxs.items():
        x += 1
        if x % 50 == 0:
            print(x)
        try: # need try b/c only used sample of WIDER face dataset, so need to check
        # image file from annotations exists in sample
            img = load_image(read_directory + img_file)
            img_name = re.findall('/(\w+)', img_file)[0]
            
            # save faces within bounding boxes as seperate images
            for i in range(len(bbxs)):
                bbx = bbxs[i]
                face = img[bbx[1]:bbx[1] + bbx[3], bbx[0]:bbx[0] + bbx[2]]
                face = resize(face, (32, 32), anti_aliasing = True, mode = 'constant')
                with warnings.catch_warnings(): # ignore "Possible precision loss" warning message from skimage.img_as_ubyte()
                    warnings.simplefilter('ignore')
                    # print(img_name, i, 'writing face image')
                    imageio.imwrite(''.join([write_directory, '/face/', img_name, '_face_', str(i), '.png']), img_as_ubyte(face))

            # randomly sample image to non-face images. try to get 2 non-faces for each face
            num_non_faces = len(bbxs) * 2
            samp_x_coord = random.sample(list(range(img.shape[1])), num_non_faces)
            samp_y_coord = random.sample(list(range(img.shape[0])), num_non_faces)

            for j in range(len(samp_x_coord)):
                x_lb = samp_x_coord[j]
                x_ub = samp_x_coord[j] + bbxs[0][2]
                # use the w and h of a face, rather than 32 x 32, to have variation in resolution of negative data
                y_lb = samp_y_coord[j]
                y_ub = samp_y_coord[j] + bbxs[0][3]

                not_intersect = True

                for bbx in bbxs:
                    # check if sampled box overlaps with any bounding boxes of faces
                    if ((bbx[0] <= x_lb <= bbx[0] + bbx[2] or bbx[0] <= x_ub <= bbx[0] + bbx[2]) \
                        and (bbx[1] <= y_lb <= bbx[1] + bbx[3] or bbx[1] <= y_ub <= bbx[1] + bbx[3])):
                        not_intersect = False

                if not_intersect:
                    non_face = img[y_lb:y_ub, x_lb:x_ub]
                    non_face = resize(non_face, (32, 32), anti_aliasing = True, mode = 'constant')
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        # print(img_name, j, 'writing non-face image')
                        imageio.imwrite(''.join([write_directory, '/non-face/', img_name, '_non_face_', str(j), '.png']), img_as_ubyte(non_face))
        except:
            continue

# later:
# get_data_imgs_bbxs('data/WIDER_annotations/wider_face_train_bbx_gt.txt')
if __name__ == '__main__':
    # train_imgs_bbxs = get_imgs_bbxs('data/WIDER_annotations/wider_face_train_bbx_gt.txt')
    test_imgs_bbxs = get_imgs_bbxs('data/WIDER_annotations/wider_face_val_bbx_gt.txt')
    # generate_pos_neg_data('data/WIDER_train/images/', train_imgs_bbxs, 'data/detection-train/')
    generate_pos_neg_data('data/WIDER_test/images/', test_imgs_bbxs, 'data/detection-test/')

