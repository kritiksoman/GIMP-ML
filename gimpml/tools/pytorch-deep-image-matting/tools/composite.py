# composite image with dataset from "deep image matting"

import os
import cv2
import math
import time
import shutil

root_dir = "/home/liuliang/Downloads/Combined_Dataset"
test_bg_dir = "/home/liuliang/Desktop/dataset/matting/VOCdevkit/VOC2012/JPEGImages"
train_bg_dir = "/home/liuliang/Desktop/dataset/matting/mscoco/train2017"


def my_composite(fg_names, bg_names, fg_dir, alpha_dir, bg_dir, num_bg, comp_dir):
    fg_ids = open(fg_names).readlines()
    bg_ids = open(bg_names).readlines()

    fg_cnt = len(fg_ids)
    bg_cnt = len(bg_ids)
    print(fg_cnt, bg_cnt)
    assert fg_cnt * num_bg == bg_cnt

    for i in range(fg_cnt):
        im_name = fg_ids[i].strip("\n").strip("\r")
        fg_path = os.path.join(fg_dir, im_name)
        alpha_path = os.path.join(alpha_dir, im_name)

        # print(fg_path, alpha_path)
        assert os.path.exists(fg_path)
        assert os.path.exists(alpha_path)

        fg = cv2.imread(fg_path)
        alpha = cv2.imread(alpha_path)

        # print("alpha shape:", alpha.shape, "image shape:", fg.shape)
        assert alpha.shape == fg.shape

        h, w, c = fg.shape
        base = i * num_bg
        for bcount in range(num_bg):
            bg_path = os.path.join(
                bg_dir, bg_ids[base + bcount].strip("\n").strip("\r")
            )
            print(base + bcount, fg_path, bg_path)
            assert os.path.exists(bg_path)
            bg = cv2.imread(bg_path)
            bh, bw, bc = bg.shape

            wratio = float(w) / bw
            hratio = float(h) / bh
            ratio = wratio if wratio > hratio else hratio
            if ratio > 1:
                new_bw = int(bw * ratio + 1.0)
                new_bh = int(bh * ratio + 1.0)
                bg = cv2.resize(bg, (new_bw, new_bh), interpolation=cv2.INTER_LINEAR)
            bg = bg[0:h, 0:w, :]
            # print(bg.shape)
            assert bg.shape == fg.shape
            alpha_f = alpha / 255.0
            comp = fg * alpha_f + bg * (1.0 - alpha_f)

            img_save_id = im_name[: len(im_name) - 4] + "_" + str(bcount) + ".png"
            comp_save_path = os.path.join(comp_dir, "image/" + img_save_id)
            fg_save_path = os.path.join(comp_dir, "fg/" + img_save_id)
            bg_save_path = os.path.join(comp_dir, "bg/" + img_save_id)
            alpha_save_path = os.path.join(comp_dir, "alpha/" + img_save_id)

            cv2.imwrite(comp_save_path, comp)
            cv2.imwrite(fg_save_path, fg)
            cv2.imwrite(bg_save_path, bg)
            cv2.imwrite(alpha_save_path, alpha)


def copy_dir2dir(src_dir, des_dir):
    for img_id in os.listdir(src_dir):
        shutil.copyfile(os.path.join(src_dir, img_id), os.path.join(des_dir, img_id))


def main():

    test_num_bg = 20
    test_fg_names = os.path.join(root_dir, "Test_set/test_fg_names.txt")
    test_bg_names = os.path.join(root_dir, "Test_set/test_bg_names.txt")
    test_fg_dir = os.path.join(root_dir, "Test_set/Adobe-licensed images/fg")
    test_alpha_dir = os.path.join(root_dir, "Test_set/Adobe-licensed images/alpha")
    test_trimap_dir = os.path.join(root_dir, "Test_set/Adobe-licensed images/trimaps")
    test_comp_dir = os.path.join(root_dir, "Test_set/comp")

    train_num_bg = 100
    train_fg_names = os.path.join(root_dir, "Training_set/training_fg_names.txt")
    train_bg_names_coco2014 = os.path.join(
        root_dir, "Training_set/training_bg_names.txt"
    )
    train_bg_names_coco2017 = os.path.join(
        root_dir, "Training_set/training_bg_names_coco2017.txt"
    )
    train_fg_dir = os.path.join(root_dir, "Training_set/all/fg")
    train_alpha_dir = os.path.join(root_dir, "Training_set/all/alpha")
    train_comp_dir = os.path.join(root_dir, "Training_set/comp")

    # change the bg names formate if is coco 2017
    fin = open(train_bg_names_coco2014, "r")
    fout = open(train_bg_names_coco2017, "w")
    lls = fin.readlines()
    for l in lls:
        fout.write(l[15:])
    fin.close()
    fout.close()

    if not os.path.exists(test_comp_dir):
        os.makedirs(test_comp_dir + "/image")
        os.makedirs(test_comp_dir + "/fg")
        os.makedirs(test_comp_dir + "/bg")
        os.makedirs(test_comp_dir + "/alpha")
        os.makedirs(test_comp_dir + "/trimap")

    if not os.path.exists(train_comp_dir):
        os.makedirs(train_comp_dir + "/image")
        os.makedirs(train_comp_dir + "/fg")
        os.makedirs(train_comp_dir + "/bg")
        os.makedirs(train_comp_dir + "/alpha")

    if not os.path.exists(train_alpha_dir):
        os.makedirs(train_alpha_dir)

    if not os.path.exists(train_fg_dir):
        os.makedirs(train_fg_dir)

    # copy test trimaps
    copy_dir2dir(test_trimap_dir, test_comp_dir + "/trimap")
    # copy train images together
    copy_dir2dir(
        os.path.join(root_dir, "Training_set/Adobe-licensed images/alpha"),
        train_alpha_dir,
    )
    copy_dir2dir(
        os.path.join(root_dir, "Training_set/Adobe-licensed images/fg"), train_fg_dir
    )
    copy_dir2dir(os.path.join(root_dir, "Training_set/Other/alpha"), train_alpha_dir)
    copy_dir2dir(os.path.join(root_dir, "Training_set/Other/fg"), train_fg_dir)

    # composite test image
    my_composite(
        test_fg_names,
        test_bg_names,
        test_fg_dir,
        test_alpha_dir,
        test_bg_dir,
        test_num_bg,
        test_comp_dir,
    )
    # composite train image
    my_composite(
        train_fg_names,
        train_bg_names_coco2017,
        train_fg_dir,
        train_alpha_dir,
        train_bg_dir,
        train_num_bg,
        train_comp_dir,
    )


if __name__ == "__main__":
    main()
