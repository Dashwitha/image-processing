# image-processing
## 1. Develop a program to display gray scale image using read and write operation.
### Grayscale Images:
Grayscale is a range of monochromatic shades from black to white. Therefore, a grayscale image contains only shades of gray and no color.While digital images can be saved as grayscale (or black and white) images, even color images contain grayscale information. This is because each pixel has a luminance value, regardless of its color. Luminance can also be described as brightness or intensity, which can be measured on a scale from black (zero intensity) to white (full intensity).Many image editing programs allow you to convert a color image to black and white, or grayscale. This process removes all color information, leaving only the luminance of each pixel.
import cv2
imgclr=cv2.imread("imgred.jpg",1)
imggry = cv2.cvtColor(imgclr, cv2.COLOR_BGR2GRAY)
cv2.imshow('imagecolor',imgclr)
cv2.imshow('imagecolor1',imggry)
cv2.imwrite('grayimg.jpg',imggry)
cv2.waitKey()

## output:

![image](https://user-images.githubusercontent.com/72288132/104895462-ec0cd900-599b-11eb-9983-924a2f7de331.png)

![image](https://user-images.githubusercontent.com/72288132/104895790-4e65d980-599c-11eb-9a9c-80888a32341c.png)

## 2. Develop a program to perform linear transformation on an image.
### linear transformation
Piece-wise Linear Transformation is type of gray level transformation that is used for image enhancement. It is a spatial domain method. It is used for manipulation of an image so that the result is more suitable than the original for a specific application.
#### a)Scaling
In computer graphics and digital imaging, image scaling refers to the resizing of a digital image. ... When scaling a raster graphics image, a new image with a higher or lower number of pixels must be generated. In the case of decreasing the pixel number (scaling down) this usually results in a visible quality loss.


## output:
![image](https://user-images.githubusercontent.com/72288132/104897222-21b2c180-599e-11eb-91dd-7416e97d4451.png) ![image](https://user-images.githubusercontent.com/72288132/104897435-5fafe580-599e-11eb-9c96-b4bd35858b3c.png)  
