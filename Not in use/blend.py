import cv2
import numpy as np
import argparse
import os
import glob
import imutils

from scipy import signal


def blur(img,if_255):


    # First a 1-D  Gaussian
    t = np.linspace(-10, 10, 300 )
    bump = np.exp(-0.1*t**2)
    bump /= np.trapz(bump) # normalize the integral to 1

    # make a 2-D kernel out of it
    kernel = bump[:, np.newaxis] * bump[np.newaxis, :]

    # # mode='same' is there to enforce the same output shape as input arrays
    # # (ie avoid border effects)
    img_cv = signal.fftconvolve(img, kernel[:, :, np.newaxis], mode='same')


    # max_val = np.max(img_cv)

    # img_cv = img_cv*255 if if_255 else img_cv
    # cv2.imshow("blurred_A",img_cv)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
 
    if not if_255: 
        return img_cv.clip(0, 255)

    return img_cv.clip(0, 255).astype(np.uint8)

def generate_GP(image_copy,gp_level,if_255):
    # gaussian_kernel = np.load('gaussian-kernel.npy')

    # G = gaussian_kernel

    gp_image = [image_copy]
    for i in range(gp_level):
        
        if not if_255: 
            # b4=image_copy.copy()
            image_copy = blur(image_copy,if_255)
            # b4_mean = np.mean(b4)
            # image_copy_mean = np.mean(image_copy)
            # cv2.imshow("image_copy",image_copy.astype(np.uint8))
            # # cv2.imshow("b4",b4.astype(np.uint8))
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

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

def blendLP(lp_A, lp_B, gp_Mask):
    A_B_pyramid = []
    n = 0
    for A_lap, B_lap in zip(lp_A, lp_B):
        n += 1
        cur_mask = gp_Mask[len(gp_Mask)-n-1]/255
        cur_A = (A_lap * (1-cur_mask)).clip(0, 255).astype(np.uint8)
        cur_B = (B_lap * cur_mask).clip(0, 255).astype(np.uint8)
        laplacian = cur_A + cur_B
        # cv2.imshow("A_lap",A_lap)
        # # cv2.imshow("b4",b4)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        A_B_pyramid.append(laplacian)

    return A_B_pyramid

def blendGP(gp_A, gp_B, gp_Mask):
    A_B_pyramid = []
    n = 0
    for A_lap, B_lap in zip(gp_A, gp_B):
        n += 1
        cur_mask = gp_Mask[n-1]/255
        cur_A = (A_lap * (1-cur_mask)).clip(0, 255).astype(np.uint8)
        cur_B = (B_lap * cur_mask).clip(0, 255).astype(np.uint8)
        laplacian = cur_A + cur_B


        # cv2.imshow("cur_B",cur_B)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        A_B_pyramid.append(laplacian)

    return A_B_pyramid

def LPB_blend (A, B, Mask,gp_level):
    # input : A, B, Mask are file directory path
    A = cv2.imread(A)
    B = cv2.imread(B)

    Mask = cv2.imread(Mask)#/255.0

    lp_level = gp_level - 1

    gp_Mask = generate_GP(Mask.copy(),gp_level,if_255=False)
    # generate Gaussian pyramid for A
    gp_A = generate_GP(A.copy(),gp_level,if_255=True)
    gp_B = generate_GP(B.copy(),gp_level,if_255=True)
    

    

    # generate Laplacian Pyramid for A
    lp_A = generate_LP(gp_A, lp_level)
    lp_B = generate_LP(gp_B, lp_level)

    # Blend LP & GP
    A_B_GP = blendGP(gp_A, gp_B,gp_Mask)
    A_B_LP = blendLP(lp_A, lp_B,gp_Mask)
    


    # now reconstruct
    
    for i in range(0, gp_level):
        A_B_reconstruct_LP = A_B_LP[i]
        A_B_reconstruct_GP = cv2.pyrUp(A_B_GP[gp_level-i])
        A_B_reconstruct = cv2.add(A_B_reconstruct_LP, A_B_reconstruct_GP)

        # A_B_reconstruct = cv2.add(A_B_LP[i], A_B_reconstruct_GP)
        if i != gp_level -1 : A_B_reconstruct  = cv2.pyrUp(A_B_reconstruct)

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
    parser.add_argument("--gpLevel", type=int, default=6)

    args = parser.parse_args()
    process(args)

