# LPB
This is a small tool for blending two face images together using the Laplacian Pyramid Blending technique.

# Data:
The test image has been included in the LPB-TestData folder. 

# Demo:
The demo output is included in the result folder, seperated by different gpLevel.

# How to use:
`python blend.py` Output images wil be saved in `result` directory

Two optional parameters:

`--folder`  : [String] the folder for the input images. The code will look for subdirectory A and B under the folder. Note that images in folder A contributes to the background, and folder B images contribute to the face. 

`--gpLevel` : [Int]  This is the parameter allowing users to change how many times we down-scale the input images for the blend. From the experiment results, more downscaling will have a smoother blend but a brighter overall tint.  


# Compare to baseline [Left: ours | Right: Baseline]

![Compare to baseline](https://github.com/mikelmh025/LPB/blob/main/vsBaseline/example1.png)

![Compare to baseline](https://github.com/mikelmh025/LPB/blob/main/vsBaseline/example2.png)

![Compare to baseline](https://github.com/mikelmh025/LPB/blob/main/vsBaseline/example3.png)
