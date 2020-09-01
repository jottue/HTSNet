import os
import glob
import math
from tqdm import tqdm
import argparse
import numpy as np
import cv2
from scipy.misc import imread



# ---------------------------
# ----- SET YOUR PATH!! -----
# ---------------------------
DATA_ROOT = './dataset'
SAVE_DIR = './'

NUM_SENTENCE = 100
SCALES = [0.7, 1., 1.5]
PATCH_SIZE=128
STRIDE_SCALE = 1.5


parser = argparse.ArgumentParser(description="synthesis_diagram")
parser.add_argument('--data_root', type= str, default= DATA_ROOT)
parser.add_argument('--save_dir',type=str, default= SAVE_DIR)
parser.add_argument('--patch_size',type=int, default=PATCH_SIZE)
parser.add_argument('--stride_scale', type=float, default=STRIDE_SCALE)

args = parser.parse_args()

argd = args.__dict__
print("Arguments:")

for key, value in argd.items():
    print('\t%15s:\t%s' % (key, value))


scribs = glob.glob(os.path.join(args.data_root,'IAM/*/*/*.png'))
docs = glob.glob(os.path.join(args.data_root,'Documents/*/*.png'))

for cate in ['syn','label']:
    path_ = os.path.join(args.save_dir,cate)
    if not os.path.exists(path_):
        os.makedirs(path_)


def otsu_b(img, kernel=3):
    blur = cv2.GaussianBlur(img, (kernel, kernel), 0)
    _, otsu_image = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return otsu_image


def make_sentence_paper(sentence_files, target_shape, max_rot = 20):
    max_rotate = max_rot
    default_image = np.zeros(target_shape)
    for x in sentence_files:
        st = imread(x, mode='L')
        inverse_st = 255. - st
        y_t = np.random.random_integers(np.int32(target_shape[1]) * -0.3, np.int32(target_shape[1]) * 0.5)
        x_t = np.random.random_integers(np.int32(target_shape[0]) * 0.005, np.int32(target_shape[0]) * 0.95)
        T = np.float32([[1, 0, y_t],
                        [0, 1, x_t]])
        translated_inverse_st = cv2.warpAffine(inverse_st, T, (target_shape[1], target_shape[0]))
        R_Matrix = cv2.getRotationMatrix2D((y_t, x_t), np.clip(np.random.randn() * 5, -max_rotate, max_rotate),
                                    np.random.uniform(0.3,1.7))
        img_w = cv2.warpAffine(translated_inverse_st, R_Matrix, (target_shape[1], target_shape[0]))

        default_image += img_w

        hand = np.uint8(np.clip(255 - default_image, 0, 255))
        if np.count_nonzero(otsu_b(255-hand))/(target_shape[0]*target_shape[1]) > 0.1:
            break

    return hand



def make_images(doc, scrib_files, inter=True):
    hand = make_sentence_paper(scrib_files, doc.shape)
    m_label = 255. - otsu_b(doc)
    h_label = 255. - otsu_b(hand)

    # h = (255.+np.random.randn()*30- hand) * (h_label / 255.)
    # h= np.clip(h, 0,255)

    ori_image = doc
    # distorted_image = np.uint8(255 - np.clip(((255. - m) + (h)), 0, 255))
    back_label = np.uint8(255 - np.clip(h_label + m_label , 0, 255))
    machine_label = m_label
    if inter:
        hand_label = h_label
    else:
        temp = machine_label/255. * h_label/255.
        hand_label = h_label-255*temp
    # input_label = np.concatenate((ori_image[:,:,None],hand[:,:,None]),axis=2)
    mask_label = np.concatenate((back_label[:,:,None],machine_label[:,:,None],hand_label[:,:,None]),axis=2)

    return ori_image, hand, mask_label



def synthesis_patch(input_patch, hand_patch, label_patch):
    machine = input_patch
    hand= hand_patch
    hand_label = label_patch[:,:,2]

    masked_hand = np.clip((255. + np.random.randn()*15 - hand) * (hand_label/255.),0,255)
    syn_patch = np.uint8(255 - np.clip(((255.-machine) + masked_hand),0,255))

    return syn_patch




for _ in range(10):
    np.random.shuffle(scribs)
    np.random.shuffle(docs)




saved_img = 0
for idx, doc in tqdm(enumerate(docs)):
    for ii, scale in enumerate(SCALES):
        real_num = NUM_SENTENCE
        document = cv2.resize(imread(doc, mode='L'), None, fx=scale , fy=scale, interpolation=cv2.INTER_NEAREST)
        st_file = np.random.choice(scribs, np.int32(real_num))


        ori_image, hand, mask_label = make_images(document, st_file, True)

        col_stair = np.int32(args.patch_size * args.stride_scale)
        row_stair = np.int32(args.patch_size * args.stride_scale)



        for x in range(math.ceil(ori_image.shape[0]//col_stair)):
            for y in range(math.ceil(ori_image.shape[1]//row_stair)):
                ori_patch = ori_image[col_stair * x:col_stair * x + args.patch_size, row_stair * y:row_stair * y + args.patch_size]
                hand_patch = hand[col_stair * x:col_stair * x + args.patch_size, row_stair * y:row_stair * y + args.patch_size]
                mask_patch = mask_label[col_stair * x:col_stair * x + args.patch_size, row_stair * y:row_stair * y + args.patch_size]

                if ori_patch.shape[0] == args.patch_size and ori_patch.shape[1] == args.patch_size:
                    temp = mask_patch[:,:,0]

                    ox,oy = np.gradient(ori_patch)
                    gx, gy = np.gradient(mask_patch[:, :, 1])
                    tem_o = np.mean(abs(ox)) + np.mean(abs(oy))
                    tempp = np.mean(abs(gx)) + np.mean(abs(gy))



                    if tempp > 5 :
                        if np.mean(temp) < 230.and np.mean(temp) > 50:
                            syn_patch = synthesis_patch(ori_patch,hand_patch, mask_patch)
                            basename = "{}_{}_{}_{}.png".format(idx,scale,ii,x * math.ceil(ori_image.shape[1] // row_stair) + y)
                            file_name = os.path.join(os.path.join(args.save_dir,'syn'), basename)
                            cv2.imwrite(file_name,  syn_patch)
                            cv2.imwrite(file_name.replace('syn', 'label'), mask_patch)

                            saved_img += 1

    print("{} images are saved".format(saved_img))

