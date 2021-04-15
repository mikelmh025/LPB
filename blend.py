import cv2
import numpy as np
import argparse
import os
import glob

def generate_GP(image_copy,gp_level):
    gp_image = [image_copy]
    for i in range(gp_level):
        image_copy = cv2.pyrDown(image_copy)
        gp_image.append(image_copy)
    return gp_image

def generate_LP(gp, lp_level):
    image_copy = gp[lp_level]
    lp_image = [image_copy]
    for i in range(lp_level, 0, -1):
        gaussian_expanded = cv2.pyrUp(gp[i])
        laplacian = cv2.subtract(gp[i-1], gaussian_expanded)
        lp_image.append(laplacian)
    return lp_image

def blend(lp_A, lp_B,gp_Mask):
    A_B_pyramid = []
    n = 0
    for A_lap, B_lap in zip(lp_A, lp_B):
        n += 1
        cols, rows, ch = A_lap.shape
        test_mask = gp_Mask[len(gp_Mask)-n-1]
        test_A = (A_lap * (1-test_mask)).clip(0, 255).astype(np.uint8)
        test_B = (B_lap * test_mask).clip(0, 255).astype(np.uint8)
        laplacian = test_A + test_B
        A_B_pyramid.append(laplacian)

    return A_B_pyramid

def LPB_blend (A, B, Mask,gp_level):
    # input : A, B, Mask are file directory path
    A = cv2.imread(A)
    B = cv2.imread(B)
    Mask = cv2.imread(Mask)/255.0

    lp_level = gp_level - 1

    # generate Gaussian pyramid for A
    gp_A = generate_GP(A.copy(),gp_level)
    gp_B = generate_GP(B.copy(),gp_level)
    gp_Mask = generate_GP(Mask.copy(),gp_level)

    # generate Laplacian Pyramid for A
    lp_A = generate_LP(gp_A, lp_level)
    lp_B = generate_LP(gp_B, lp_level)

    # Blend
    A_B_pyramid = blend(lp_A, lp_B,gp_Mask)

    # now reconstruct
    A_B_reconstruct = A_B_pyramid[0]
    for i in range(1, gp_level):
        A_B_reconstruct = cv2.pyrUp(A_B_reconstruct)
        A_B_reconstruct = cv2.add(A_B_pyramid[i], A_B_reconstruct)

    return A_B_reconstruct
    

def process (args):
    save_dir = os.path.join("result","gpLevel="+str(args.gpLevel))
    os.makedirs(save_dir, exist_ok=True)

    for filename in glob.glob(os.path.join(args.folder, "A","*.png")):
        # Process the filenames
        num = filename.split("/")[-1].split(".")[-2].split("_")[-1]
        name_B    = "pixar_"+str(num).zfill(6)+".png"
        name_Mask = "pixar_"+str(num).zfill(6)+"_blend_alpha.png"
        
        A         = filename
        B         = os.path.join(args.folder, "B"     ,name_B   )
        Mask      = os.path.join(args.folder, "Mask"  ,name_Mask)
        save_path = os.path.join(save_dir,name_Mask)
        # Blend Image A and B
        output = LPB_blend (A, B, Mask, args.gpLevel)
        cv2.imwrite(os.path.join(save_path),output)



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default="LPB-TestData")
    parser.add_argument("--gpLevel", type=str, default=3)

    args = parser.parse_args()
    process(args)

