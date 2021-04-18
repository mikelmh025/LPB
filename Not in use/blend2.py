import cv2
import numpy as np
import argparse
import os
import glob
import imutils

from scipy import signal


def blur(img):


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
 
    
    return img_cv.clip(0, 255)

    return img_cv.clip(0, 255).astype(np.uint8)

def one_level_laplacian(img):
    # generate Gaussian pyramid for Apple
    A = img.copy()


    # Downsample blurred A
    small_A = cv2.pyrDown(A)

    # Upsample small, blurred A
    # insert zeros between pixels, then apply a gaussian low pass filter
    upsampled_A = cv2.pyrUp(small_A)

    # generate Laplacian level for A
    laplace_A = A -upsampled_A# cv2.subtract(A , upsampled_A)
    # laplace_A = cv2.subtract(A , upsampled_A)

    
    return small_A, upsampled_A, laplace_A

def generate_GP(image_copy,gp_level):

    gp_image = [image_copy]
    for i in range(gp_level):
        
        # image_copy = blur(image_copy)

        image_copy = cv2.pyrDown(image_copy)
        gp_image.append(image_copy)
        # cv2.imshow("image_copy",image_copy)#.clip(0, 255).astype(np.uint8))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
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
    # input : A, B, Mask are file directory path
    A = cv2.imread(A)#[:,:,0]
    B = cv2.imread(B)#[:,:,0]
    Mask    = cv2.imread(Mask)#[:,:,0]
    Mask_neg = cv2.bitwise_not(Mask)
    

    # generate Gaussian pyramid for A
    gp_A,lp_A   = generate_GP_LP(A.copy()       ,gp_level)    
    gp_B,lp_B   = generate_GP_LP(B.copy()       ,gp_level)
    gp_Mask     = generate_GP   (Mask.copy()    ,gp_level)    
    gp_Mask_neg = generate_GP   (Mask_neg.copy(),gp_level)  



    curr_mask     = gp_Mask[-1]     /255
    curr_mask_neg = gp_Mask_neg[-1] /255


    start_A = gp_A[-1] * (curr_mask_neg)
    start_B = gp_B[-1] * (  curr_mask)
    start_A = start_A.astype(np.uint8)
    start_B = start_B.astype(np.uint8)
    start   = start_A + start_B

    cv2.imshow("curr_mask",curr_mask)
    cv2.imshow("start",start)#.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    for i in reversed(range(0, gp_level)):
        curr_mask     = gp_Mask[i]     /255
        curr_mask_neg = gp_Mask_neg[i] /255
        lp_A_masked = lp_A[i] * curr_mask_neg .astype(np.uint8)

        lp_B_masked = lp_B[i] * curr_mask     .astype(np.uint8)

        LP_S = lp_A_masked + lp_B_masked
        reconstructed_A = cv2.pyrUp(start)
        reconstructed_A = reconstructed_A+ LP_S #cv2.add(reconstructed_A, LP_S)
        
        cv2.imshow("curr_mask",curr_mask)
        cv2.imshow("start",cv2.pyrUp(start))
        cv2.imshow("LP_S",LP_S)
        cv2.imshow("reconstructed_A",reconstructed_A)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        start = reconstructed_A.astype(np.uint8)

    reconstruct = reconstructed_A
    # cv2.imshow("reconstruct",reconstructed_A.clip(0, 255).astype(np.uint8))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return reconstruct
    

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
    parser.add_argument("--gpLevel", type=int, default=2)

    args = parser.parse_args()
    process(args)

