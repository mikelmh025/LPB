import numpy as np
import matplotlib.pyplot as plt

from skimage import data
from skimage.transform import pyramid_gaussian,pyramid_laplacian,pyramid_reduce,pyramid_expand





import cv2
import numpy as np
import argparse
import os
import glob
import imutils

from scipy import signal


def blur(img):

    img = img *255

    # First a 1-D  Gaussian
    t = np.linspace(-10, 10, 30 )
    bump = np.exp(-0.1*t**2)
    bump /= np.trapz(bump) # normalize the integral to 1

    # make a 2-D kernel out of it
    kernel = bump[:, np.newaxis] * bump[np.newaxis, :]

    # # mode='same' is there to enforce the same output shape as input arrays
    # # (ie avoid border effects)
    img_cv = signal.fftconvolve(img, kernel[:, :, np.newaxis], mode='same')

    img_cv = img_cv/255
    
    return img_cv.clip(0, 1)


def one_level_laplacian(img):
    # generate Gaussian pyramid for Apple
    A = img.copy()


    # Downsample blurred A
    small_A = pyramid_reduce(A,downscale=2, multichannel=True)

    # Upsample small, blurred A
    # insert zeros between pixels, then apply a gaussian low pass filter
    upsampled_A = pyramid_expand(small_A,upscale=2, multichannel=True)

    # generate Laplacian level for A
    laplace_A = A -upsampled_A# cv2.subtract(A , upsampled_A)
    # laplace_A = cv2.subtract(A , upsampled_A)

    
    return small_A, upsampled_A, laplace_A

def generate_GP(image_copy,gp_level):

    gp_image = [image_copy]
    for i in range(gp_level):
        
        # image_copy = blur(image_copy)

        image_copy = pyramid_reduce(image_copy,downscale=2, multichannel=True,sigma= 3)
        gp_image.append(image_copy)
        
    return gp_image

def generate_GP_LP(image_copy,gp_level):

    img = image_copy
    gp_image = [img]
    lp_image = []
    F_upsampled = []
    for i in range(gp_level):
        small_A, upsampled_A, laplace_A = one_level_laplacian(img)
        gp_image.append(small_A)
        F_upsampled.append(upsampled_A)
        lp_image.append(laplace_A)
        img = small_A

        
    return gp_image, lp_image



def LPB_blend (A, B, Mask,gp_level):

    # image = data.astronaut()
    A = plt.imread(A)
    B = plt.imread(B)/255
    Mask = plt.imread(Mask)
    Mask_neg = 1 - Mask

    GP_A,LP_A   = generate_GP_LP(A.copy()       ,gp_level)    
    GP_B,LP_B   = generate_GP_LP(B.copy()       ,gp_level)
    GP_Mask     = generate_GP   (Mask.copy()    ,gp_level)    
    fig, ax = plt.subplots()
    ax.imshow(GP_Mask[2])
    plt.savefig("out.png")

    GP_Mask_neg = generate_GP   (Mask_neg.copy(),gp_level)  
    

    start_A = GP_A[gp_level] * (GP_Mask_neg[gp_level])
    start_B = GP_B[gp_level] * (GP_Mask[gp_level])
    start = start_A + start_B

    for i in reversed(range(0, gp_level)):
        curr_mask     = GP_Mask[i]     
        curr_mask_neg = GP_Mask_neg[i] 
        lp_A_masked = LP_A[i] * curr_mask_neg 
        lp_B_masked = LP_B[i] * curr_mask     
        LP_S = lp_A_masked  + lp_B_masked

        reconstructed = pyramid_expand(start,upscale=2, multichannel=True) + LP_S
        start = reconstructed


    return reconstructed
    

def process (args):
    save_dir = os.path.join("result","gpLevel="+str(args.gpLevel))
    os.makedirs(save_dir, exist_ok=True)

    for filename in glob.glob(os.path.join(args.folder, "A","*.png")):
        # Process the filenames
        num = filename.split("/")[-1].split(".")[-2].split("_")[-1]
        name_B    = str(num).zfill(6)+"_pixar"+".jpg"
        name_Mask = str(num).zfill(6)+"_pixar_alpha.png"
        
        A         = filename
        B         = os.path.join(args.folder, "B"     ,name_B   )
        Mask      = os.path.join(args.folder, "Mask"  ,name_Mask)
        save_path = os.path.join(save_dir,name_Mask)
        # Blend Image A and B
        output = LPB_blend (A, B, Mask, args.gpLevel)
        fig, ax = plt.subplots()
        ax.imshow(output)
        plt.savefig(os.path.join(save_path))
        # cv2.imwrite(os.path.join(save_path),output)



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default="TestData")
    parser.add_argument("--gpLevel", type=int, default=6)

    args = parser.parse_args()
    process(args)

