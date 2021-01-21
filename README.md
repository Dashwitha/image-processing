# image-processing
## 1. Develop a program to display gray scale image using read and write operation.
### Grayscale Images:
Grayscale is a range of monochromatic shades from black to white. Therefore, a grayscale image contains only shades of gray and no color.While digital images can be saved as grayscale (or black and white) images, even color images contain grayscale information. This is because each pixel has a luminance value, regardless of its color. Luminance can also be described as brightness or intensity, which can be measured on a scale from black (zero intensity) to white (full intensity).Many image editing programs allow you to convert a color image to black and white, or grayscale. This process removes all color information, leaving only the luminance of each pixel.
### code
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
### code
import cv2 
imgclr=cv2.imread("imgred.jpg") 
res = cv2.resize(imgclr,(300,300),interpolation=cv2.INTER_CUBIC) 
cv2.imshow('imagecolor',imgclr)
cv2.imshow('imagecolor1',res)
cv2.waitKey()


## output:
![image](https://user-images.githubusercontent.com/72288132/104897222-21b2c180-599e-11eb-91dd-7416e97d4451.png)   ![image](https://user-images.githubusercontent.com/72288132/104897435-5fafe580-599e-11eb-9c96-b4bd35858b3c.png)  
 
 ## b)Rotation
 Image rotation is a common image processing routine with applications in matching, alignment, and other image-based algorithms. ... An image rotated by 45°. The output is the same size as the input, and the out of edge values are dropped.
 ### code
 import cv2 
imgclr=cv2.imread("colorimg.jpg") 
(row, col) = imgclr.shape[:2] 
M = cv2.getRotationMatrix2D((col / 2, row/ 2), 45, 1)
res = cv2.warpAffine(imgclr, M, (col,row)) 
cv2.imshow('imagecolor',imgclr)
cv2.imshow('imagecolor1',res)
cv2.waitKey()
## output:

![image](https://user-images.githubusercontent.com/72288132/104898026-1f9d3280-599f-11eb-99c5-de20435777f7.png)     ![image](https://user-images.githubusercontent.com/72288132/104898088-2f1c7b80-599f-11eb-9648-6348bd752e86.png)
 ## 3.Develop a program to find the sum and mean of a set of images
 In digital image processing, the sum of absolute differences (SAD) is a measure of the similarity between image blocks. It is calculated by taking the absolute difference between each pixel in the original block and the corresponding pixel in the block being used for comparison. These differences are summed to create a simple metric of block similarity, the L1 norm of the difference image or Manhattan distance between two image blocks.
 #### code
sum and mean
import cv2
import glob 
import numpy as np
from PIL import Image
path=glob.glob("E:\pics\*.jpg")
for file in path:
    print(file)
    image=cv2.imread(file)
    sum=image+sum
mean=sum/20
cv2.imshow("Sum",sum)
cv2.waitKey(0)
cv2.imshow("Mean",mean)
cv2.waitKey(0)
cv2.destroyAllWindows()

## output:
![image](https://user-images.githubusercontent.com/72288132/104898835-1e203a00-59a0-11eb-8489-0401223cdde9.png)
![image](https://user-images.githubusercontent.com/72288132/104898884-2d06ec80-59a0-11eb-8e09-632f25110a1c.png)

## 4. Convert color image gray scale to binary image
Thresholding is the simplest method of image segmentation and the most common way to convert a grayscale image to a binary image. ... Here g(x, y) represents threshold image pixel at (x, y) and f(x, y) represents greyscale image pixel at (x, y).

### code
import cv2 
img=cv2.imread("nature.jpg")
cv2.imshow("Original Image",img)
gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray images',gray)  
cv2.waitKey(0) 
ret,bw_img= cv2.threshold(img, 127, 255, cv2.THRESH_BINARY) 
cv2.imshow("Binary",bw_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
#### output:
![image](https://user-images.githubusercontent.com/72288132/104899629-19a85100-59a1-11eb-8f87-3311cac70189.png)
![image](https://user-images.githubusercontent.com/72288132/104899644-1f9e3200-59a1-11eb-93e3-46674a50bea8.png)
![image](https://user-images.githubusercontent.com/72288132/104899659-23ca4f80-59a1-11eb-806f-04385f863b3f.png)

## 5. convert a color image to different color space
Color spaces are different types of color modes, used in image processing and signals and system for various purposes. Some of the common color spaces are:

RGB CMY’K Y’UV YIQ Y’CbCr HSV Color space conversion is the translation of the representation of a color from one basis to another. This typically occurs in the context of converting an image that is represented in one color space to another color space, the goal being to make the translated image look as similar as possible to the original.
### code
import cv2 
image=cv2.imread("nature.jpg")
cv2.imshow("Original Image",image)
cv2.waitKey(0)
gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray images',gray)  
cv2.waitKey(0) 
YCrCb=cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb) 
cv2.imshow('YCrCb image',YCrCb)
cv2.waitKey(0)
HSV=cv2.cvtColor(image, cv2.COLOR_BGR2HSV) 
cv2.imshow('HSV image',HSV)
cv2.waitKey(0)          
cv2.destroyAllWindows()
#### output:


![image](https://user-images.githubusercontent.com/72288132/104899990-86235000-59a1-11eb-91a1-595c2695178e.png)
![image](https://user-images.githubusercontent.com/72288132/104899997-89b6d700-59a1-11eb-8d5c-d34f069b0769.png)
![image](https://user-images.githubusercontent.com/72288132/104900017-8e7b8b00-59a1-11eb-8042-f435f70129c7.png)
![image](https://user-images.githubusercontent.com/72288132/104900030-94716c00-59a1-11eb-9399-3aae4e68b2a0.png)

## 6. Develop a program to create an image from 2D array generate an array of random size.
A digital image is nothing more than data—numbers indicating variations of red, green, and blue at a particular location on a grid of pixels. Most of the time, we view these pixels as miniature rectangles sandwiched together on a computer screen. With a little creative thinking and some lower level manipulation of pixels with code, however, we can display that information in a myriad of ways. This tutorial is dedicated to breaking out of simple shape drawing in Processing and using images (and their pixels) as the building blocks of Processing graphics.
### code
from PIL import Image
import numpy as np
w, h = 512, 512
data = np.zeros((h, w, 3), dtype=np.uint8)
data[0:256, 0:256] = [255, 80, 20] # red patch in upper left
img = Image.fromarray(data, 'RGB')
img.save('my.png')
img.show()
#### output:
![image](https://user-images.githubusercontent.com/72288132/104900394-09dd3c80-59a2-11eb-9b52-aff01f407fdf.png)

## 7. Find the neighbors matrix.
### The numpy.zeros() function returns a new array of given shape and type, with zeros.append () Syntax: list_name.append (‘value’) It takes only one argument. This function appends the incoming element to the end of the list as a single new element.
### code
import numpy as np
M = [[1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]]
M = np.asarray(M)
N = np.zeros(M.shape)
def sumNeighbors(M,x,y):
    l = []
    for i in range(max(0,x-1),x+2):
        for j in range(max(0,y-1),y+2):
            try:
                t = M[i][j]
                l.append(t)
            except IndexError:
                pass
    return sum(l)-M[x][y]
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        N[i][j] = sumNeighbors(M, i, j)
print ("Original matrix:\n", M)
print ("Summed neighbors matrix:\n", N)

### output:

Original matrix:
 [[1 2 3]
 
 [4 5 6]
 
 [7 8 9]]
 
Summed neighbors matrix:

 [[11. 19. 13.]
 
 [23. 40. 27.]
 
 [17. 31. 19.]]


## 8)Write a c++ program to perform operator overloading
### code
 #include <iostream>
using namespace std;
class matrix
{
 int r1, c1, i, j, a1;
 int a[10][10];

public:int get()
 {
  cout << "Enter the row and column size for the  matrix\n";
  cin >> r1;
  cin >> c1;
   cout << "Enter the elements of the matrix\n";
  for (i = 0; i < r1; i++)
  {
   for (j = 0; j < c1; j++)
   {
    cin>>a[i][j];

   }
  }
 
 
 };
 void operator+(matrix a1)
 {
 int c[i][j];
 
   for (i = 0; i < r1; i++)
   {
    for (j = 0; j < c1; j++)
    {
     c[i][j] = a[i][j] + a1.a[i][j];
    }
   
  }
  cout<<"addition is\n";
  for(i=0;i<r1;i++)
  {
   cout<<" ";
   for (j = 0; j < c1; j++)
   {
    cout<<c[i][j]<<"\t";
   }
   cout<<"\n";
  }

 };

  void operator-(matrix a2)
 {
 int c[i][j];
 
   for (i = 0; i < r1; i++)
   {
    for (j = 0; j < c1; j++)
    {
     c[i][j] = a[i][j] - a2.a[i][j];
    }
   
  }
  cout<<"subtraction is\n";
  for(i=0;i<r1;i++)
  {
   cout<<" ";
   for (j = 0; j < c1; j++)
   {
    cout<<c[i][j]<<"\t";
   }
   cout<<"\n";
  }
 };

 void operator*(matrix a3)
 {
  int c[i][j];

  for (i = 0; i < r1; i++)
  {
   for (j = 0; j < c1; j++)
   {
    c[i][j] =0;
    for (int k = 0; k < r1; k++)
    {
     c[i][j] += a[i][k] * (a3.a[k][j]);
    }
  }
  }
  cout << "multiplication is\n";
  for (i = 0; i < r1; i++)
  {
   cout << " ";
   for (j = 0; j < c1; j++)
   {
    cout << c[i][j] << "\t";
   }
   cout << "\n";
  }
 };

};

int main()
{
 matrix p,q;
 p.get();
 q.get();
 p + q;
 p - q;
 p * q;
return 0;
}

### Output:

Enter the row and column size for the  matrix

2

2

Enter the elements of the matrix

9

8

7

6

Enter the row and column size for the  matrix

2

2

Enter the elements of the matrix

5

4

3

2

addition is

 14      12
 
 10      8
 
subtraction is

 4      4
 
 4      4
 
multiplication is

 69     52
 
 53     40



## Develop a program to implement negative transformation of an image

The output of image inversion is a negative of a digital image.
In a digital image the intensity levels vary from 0 to L-1. The negative transformation is given by s=L-1-r.
When an image is inverted, each of its pixel value ‘r’ is subtracted from the maximum pixel value L-1 and the original pixel is replaced with the result ‘s’.
Image inversion or Image negation helps finding the details from the darker regions of the image.
### code
import cv2
img = cv2.imread('flower.jpg')
cv2.imshow('Original',img)
cv2.waitKey(0)
neg=255-img
cv2.imshow('negetive',neg)
cv2.waitKey(0);
cv2.destroyAllWindows()
#### output
![image](https://user-images.githubusercontent.com/72288132/105325627-ad387800-5b81-11eb-96d2-9f4d772273d9.png)    ![image](https://user-images.githubusercontent.com/72288132/105325686-b9bcd080-5b81-11eb-8af3-748579a88c8c.png)


## Develop a program to implement contrast of an image 

Contrast is created by the difference in luminance reflected from two adjacent surfaces. In other words, contrast is the difference in visual properties that makes an object distinguishable from other objects and the background. Contrast is an important factor in any subjective evaluation of image quality.

### code 
from PIL import Image, ImageEnhance
img = Image.open("flower.jpg")
img.show()
img=ImageEnhance.Color(img)
img.enhance(2.0).show()
### output
![image](https://user-images.githubusercontent.com/72288132/105328855-57fe6580-5b85-11eb-9e9e-5d2be06bd699.png)
![image](https://user-images.githubusercontent.com/72288132/105329103-a27fe200-5b85-11eb-9cf2-be83c210f488.png)

