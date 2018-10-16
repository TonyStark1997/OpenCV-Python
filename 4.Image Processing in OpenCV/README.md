# 第四章：OpenCV中的图像处理

本章节你将学习图像的改变色彩空间、提取对象、图像的几何变换、图像的阈值、平滑图像等。

更多内容请关注我的GitHub库：https://github.com/TonyStark1997，如果喜欢，star并follow我！

***

## 一、改变色彩空间

***

### 目标：

本章节您需要学习以下内容:

*在本教程中，你将学习如何将图像从一个颜色空间转换为另一个颜色空间，例如BGR↔Gray，BGR↔HSV等。
*除此之外，我们还将创建一个提取视频中彩色对象的应用程序
*你将学习以下函数：cv.cvtColor（），cv.inRange（）等。

### 1.改变色彩空间

OpenCV中有150多种颜色空间转换方法。但我们将只研究两种最广泛使用的转换方法，BGR↔Gray和BGR↔HSV。

对于颜色转换，我们使用函数cv.cvtColor（input_image，flag），其中flag确定转换类型。

对于BGR→Gray转换，我们使用标志cv.COLOR_BGR2GRAY。类似地，对于BGR→HSV，我们使用标志cv.COLOR_BGR2HSV。要获取其他标志，只需在Python终端中运行以下命令：

```python
>>> import cv2 as cv
>>> flags = [i for i in dir(cv) if i.startswith('COLOR_')]
>>> print( flags )
```

**注意：对于HSV，色调范围是[0,179]，饱和范围是[0,255]，值范围是[0,255]。不同的软件使用不同的规模，因此，如果要将OpenCV值与它们进行比较，则需要对这些范围进行标准化。**

### 2.对象提取

现在我们知道如何将BGR图像转换为HSV，我们可以使用它来提取彩色对象。在HSV中表示颜色比在BGR颜色空间中更容易。在我们的应用程序中，我们将尝试提取蓝色对象。 所以这是方法：

* 拍摄视频的每一帧
* 从BGR转换为HSV颜色空间
* 将HSV图像阈值设为一系列蓝色
* 现在单独提取蓝色对象，我们可以对我们想要的图像做任何事情。

以下是详细评论的代码：

```python
import cv2 as cv
import numpy as np
cap = cv.VideoCapture(0)
while(1):
# Take each frame
_, frame = cap.read()
# Convert BGR to HSV
hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
# define range of blue color in HSV
lower_blue = np.array([110,50,50])
upper_blue = np.array([130,255,255])
# Threshold the HSV image to get only blue colors
mask = cv.inRange(hsv, lower_blue, upper_blue)
# Bitwise-AND mask and original image
res = cv.bitwise_and(frame,frame, mask= mask)
cv.imshow('frame',frame)
cv.imshow('mask',mask)
cv.imshow('res',res)
k = cv.waitKey(5) & 0xFF
if k == 27:
break
cv.destroyAllWindows()
```

下面的图片展示了我们提取蓝色对象后的效果：

![image1](https://raw.githubusercontent.com/TonyStark1997/OpenCV-Python/master/4.Image%20Processing%20in%20OpenCV/Image/image1.png)

**注意：图像中有一些噪音。 我们将在后面的章节中看到如何删除它们。这是对象提取中最简单的方法。 一旦你学习了轮廓的功能，你就可以做很多事情，比如找到这个物体的质心并用它来追踪物体，只需在镜头前移动你的手以及许多其他有趣的东西来绘制图表。**

### 3.如何找到要跟踪的HSV值

这是stackoverflow.com中常见的问题。它非常简单，你可以使用相同的函数cv.cvtColor（）。您只需传递所需的BGR值，而不是传递图像。例如，要查找绿色的HSV值，在Python终端中输入以下命令：

```python
>>> green = np.uint8([[[0,255,0 ]]])
>>> hsv_green = cv.cvtColor(green,cv.COLOR_BGR2HSV)
>>> print( hsv_green )
[[[ 60 255 255]]]
```

现在分别将[H-10,100,100]和[H + 10,255,255]作为下限和上限。除了这种方法，你可以使用任何图像编辑工具，如GIMP或任何在线转换器来查找这些值，但不要忘记调整HSV范围。

## 二、图像的几何变换

***

### 目标

本章节您需要学习以下内容:

*学习将不同的几何变换应用于图像，如平移，旋转，仿射变换等。
*你将看到以下函数：cv.getPerspectiveTransform

### 1.转换

OpenCV提供了两个转换函数cv.warpAffine和cv.warpPerspective，你可以使用它们进行各种转换。cv.warpAffine采用2x3变换矩阵作为输入，而cv.warpPerspective采用3x3变换矩阵作为输入。

### 2.缩放

缩放只是调整图像大小。为此，OpenCV附带了一个函数cv.resize（）。可以手动指定图像的大小，也可以指定缩放系数。可以使用不同的插值方法，常用的插值方法是用于缩小的cv.INTER_AREA和用于缩放的cv.INTER_CUBIC（慢）和cv.INTER_LINEAR。默认情况下，使用的插值方法是cv.INTER_LINEAR，它用于所有调整大小的目的。你可以使用以下方法之一调整输入图像的大小：

```python
import numpy as np
import cv2 as cv
img = cv.imread('messi5.jpg')
res = cv.resize(img,None,fx=2, fy=2, interpolation = cv.INTER_CUBIC)
#OR
height, width = img.shape[:2]
res = cv.resize(img,(2*width, 2*height), interpolation = cv.INTER_CUBIC)
```
### 3.翻译

翻译是对象位置的位移。如果你知道像素点（x，y）要位移的距离，让它为变为（$t_x$,$t_y$），你可以创建变换矩阵**M**，如下所示：

$$M=\begin{bmatrix}
1&0&t_x\\
0&1&t_y\\
\end{bmatrix}$$

你可以将其设置为np.float32类型的Numpy数组，并将其传递给cv.warpAffine（）函数。下面的示例演示图像像素点整体进行（100,50）位移：

```python
import numpy as np
import cv2 as cv
img = cv.imread('messi5.jpg',0)
rows,cols = img.shape
M = np.float32([[1,0,100],[0,1,50]])
dst = cv.warpAffine(img,M,(cols,rows))
cv.imshow('img',dst)
cv.waitKey(0)
cv.destroyAllWindows()
```

**注意：cv.warpAffine（）函数的第三个参数是输出图像的大小，它应该是（宽度，高度）的形式。请记住 width=列数 ，height=行数。**

请看下面的结果：

![image2](https://raw.githubusercontent.com/TonyStark1997/OpenCV-Python/master/4.Image%20Processing%20in%20OpenCV/Image/image2.png)

### 4.旋转

通过改变图像矩阵实现图像旋转角度θ

$$M=\begin{bmatrix}
cos\Theta &-sin\Theta\\ 
sin\Theta & cos\Theta 
\end{bmatrix}$$

但OpenCV提供可调旋转，即旋转中心可调，因此您可以在任何喜欢的位置旋转。修正的变换矩阵由下式给出：

$$M=\begin{bmatrix}
\alpha  & \beta & \left ( 1-\alpha  \right )\cdot center.x-\beta \cdot center.y \\ 
-\beta   & \alpha & \beta \cdot center.x\left ( 1-\alpha  \right )\cdot center.y
\end{bmatrix}$$

其中：

$$\alpha = scale\cdot cos\Theta$$

$$\beta = scale\cdot sin\Theta $$

为了找到这个转换矩阵，OpenCV提供了一个函数cv.getRotationMatrix2D。以下示例将图像相对于中心旋转90度而不进行任何缩放。

```python
img = cv.imread('messi5.jpg',0)
rows,cols = img.shape
M = cv.getRotationMatrix2D((cols/2,rows/2),90,1)
dst = cv.warpAffine(img,M,(cols,rows))
```

窗口将如下图显示：

![image3](https://raw.githubusercontent.com/TonyStark1997/OpenCV-Python/master/4.Image%20Processing%20in%20OpenCV/Image/image3.png)

### 5.仿射变换

在仿射变换中，原始图像中的所有平行线仍将在输出图像中平行。为了找到变换矩阵，我们需要输入图像中的三个点及其在输出图像中的相应位置。然后cv.getAffineTransform将创建一个2x3矩阵，该矩阵将传递给cv.warpAffine。

参考以下示例，并查看我选择的点（以绿色标记）：

```python
img = cv.imread('drawing.png')
rows,cols,ch = img.shape
pts1 = np.float32([[50,50],[200,50],[50,200]])
pts2 = np.float32([[10,100],[200,50],[100,250]])
M = cv.getAffineTransform(pts1,pts2)
dst = cv.warpAffine(img,M,(cols,rows))
plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()
```

窗口将如下图显示：

![image4](https://raw.githubusercontent.com/TonyStark1997/OpenCV-Python/master/4.Image%20Processing%20in%20OpenCV/Image/image4.png)

### 6.透视转型

对于透视变换，你需要一个3x3变换矩阵。即使在转换之后，直线仍将保持笔直。要找到此变换矩阵，输入图像上需要4个点，输出图像上需要相应的点。在这4个点中，其中3个不应该共线。然后可以通过函数cv.getPerspectiveTransform找到变换矩阵，将cv.warpPerspective应用于此3x3变换矩阵。

请参阅以下代码：

```python
img = cv.imread('sudoku.png')
rows,cols,ch = img.shape
pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])
M = cv.getPerspectiveTransform(pts1,pts2)
dst = cv.warpPerspective(img,M,(300,300))
plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
```

窗口将如下图显示：

![image5](https://raw.githubusercontent.com/TonyStark1997/OpenCV-Python/master/4.Image%20Processing%20in%20OpenCV/Image/image5.png)

## 三、图象阈值

***

### 目标：

本章节您需要学习以下内容:

*你将学习简单的阈值处理，自适应阈值处理，Otsu的阈值处理等。
*你将学习以下函数：cv.threshold，cv.adaptiveThreshold等。

### 1.简单的阈值处理

在这里所需要做的工作是简单易懂的。如果像素值大于阈值，则为其分配一个值（可以是白色），否则为其分配另一个值（可以是黑色）。使用的函数是cv.threshold。函数第一个参数是源图像，它应该是灰度图像。第二个参数是用于对像素值进行分类的阈值。第三个参数是maxVal，它表示如果像素值大于（有时小于）阈值则要给出的值。OpenCV提供不同类型的阈值，它由函数的第四个参数决定。不同的类型是：

* cv.THRESH_BINARY
* cv.THRESH_BINARY_INV
* cv.THRESH_TRUNC
* cv.THRESH_TOZERO
* cv.THRESH_TOZERO_INV

文档清楚地解释了每种类型的含义。 请查看文档链接。

函数将获得两个输出。第一个是retavl，将在后面解释它的作用。第二个输出是我们的阈值图像。

参考以下代码：

```python
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
img = cv.imread('gradient.png',0)
ret,thresh1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
ret,thresh2 = cv.threshold(img,127,255,cv.THRESH_BINARY_INV)
ret,thresh3 = cv.threshold(img,127,255,cv.THRESH_TRUNC)
ret,thresh4 = cv.threshold(img,127,255,cv.THRESH_TOZERO)
ret,thresh5 = cv.threshold(img,127,255,cv.THRESH_TOZERO_INV)
titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
for i in xrange(6):
plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
plt.title(titles[i])
plt.xticks([]),plt.yticks([])
plt.show()
```

**注意：为了绘制多个图像，我们使用了plt.subplot（）函数。请查看Matplotlib文档以获取更多详细信息。**

窗口将如下图显示：

![image6](https://raw.githubusercontent.com/TonyStark1997/OpenCV-Python/master/4.Image%20Processing%20in%20OpenCV/Image/image6.png)

### 2.自适应阈值处理

在上一节中，我们使用全局值作为阈值。但在图像在不同区域具有不同照明条件的所有条件下可能并不好。在那种情况下，我们进行自适应阈值处理。我们希望算法计算图像的小区域的阈值，因此，我们为同一图像的不同区域获得不同的阈值，并且它为具有不同照明的图像提供了更好的处理结果。

它有三个“特殊”输入参数和一个输出参数。

**Adaptive Method** - 自适应方法，决定如何计算阈值。

* cv.ADAPTIVE_THRESH_MEAN_C：阈值是邻域的平均值。
* cv.ADAPTIVE_THRESH_GAUSSIAN_C：阈值是邻域值的加权和，其中权重是高斯窗口。

**Block Size** - 块大小，它决定了邻域的大小。

**C** - 它只是从计算的平均值或加权平均值中减去的常数。

下面的代码比较了具有不同照明的图像的全局阈值处理和自适应阈值处理：

```python
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
img = cv.imread('sudoku.png',0)
img = cv.medianBlur(img,5)
ret,th1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
th2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,\
cv.THRESH_BINARY,11,2)
th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
cv.THRESH_BINARY,11,2)
titles = ['Original Image', 'Global Thresholding (v = 127)',
'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1, th2, th3]
for i in xrange(4):
plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
plt.title(titles[i])
plt.xticks([]),plt.yticks([])
plt.show()
```

窗口将如下图显示：

![image7](https://raw.githubusercontent.com/TonyStark1997/OpenCV-Python/master/4.Image%20Processing%20in%20OpenCV/Image/image7.png)

### 3.Otsu's 二值化

在第一节中，我只告诉你另一个参数是retVal。它的作用是来进行Otsu's二值化。

在全局阈值处理中，我们使用任意值作为阈值，那么，我们如何知道我们选择的价值是好还是不好？答案是，试错法。但考虑双峰图像（简单来说，双峰图像是直方图有两个峰值的图像）我们可以将这些峰值中间的值近似作为阈值，这就是Otsu二值化的作用。简单来说，它会根据双峰图像的图像直方图自动计算阈值。（对于非双峰图像，二值化不准确。）

为此，使用了我们的cv.threshold（）函数，但是传递了一个额外的标志cv.THRESH_OTSU。对于阈值，只需传递零。然后算法找到最佳阈值并返回第二个输出retVal。如果未使用Otsu阈值，则retVal与您使用的阈值相同。

请查看以下示例。输入图像是嘈杂的图像。在第一种情况下，我将全局阈值应用为值127。在第二种情况下，我直接应用了Otsu的阈值。在第三种情况下，我使用5x5高斯内核过滤图像以消除噪声，然后应用Otsu阈值处理。了解噪声过滤如何改善结果。

```python
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
img = cv.imread('noisy2.png',0)
# global thresholding
ret1,th1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
# Otsu's thresholding
ret2,th2 = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
# Otsu's thresholding after Gaussian filtering
blur = cv.GaussianBlur(img,(5,5),0)
ret3,th3 = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
# plot all the images and their histograms
images = [img, 0, th1,
img, 0, th2,
blur, 0, th3]
titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)',
'Original Noisy Image','Histogram',"Otsu's Thresholding",
'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]
for i in xrange(3):
plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
plt.show()
```

窗口将如下图显示：

![image8](https://raw.githubusercontent.com/TonyStark1997/OpenCV-Python/master/4.Image%20Processing%20in%20OpenCV/Image/image8.png)

下面讲解了Otsu二值化的Python实现，以展示它的实际工作原理。如果你不感兴趣，可以跳过这个内容。

由于我们正在使用双峰图像，因此Otsu的算法试图找到一个阈值（t），它最小化了由关系给出的加权类内方差：

$$\sigma _{w}^{2}\left ( t \right )= q_{1}\left ( t \right )\sigma _{1}^{2}\left ( t \right )+q_{2}\left ( t \right )\sigma _{2}^{2}\left ( t \right )$$

其中：

$$q_{1\left ( t \right )}=\sum_{i=1}^{t}P\left ( i \right ) \ \ \ \ \&\ \ \ \ q_{1\left ( t \right )}=\sum_{i=1}^{I}P\left ( i \right )$$

$$\mu _{1}\left ( t \right )=\sum_{t}^{i=1}\frac{iP\left ( i \right )}{q_{1}\left ( t \right )}\ \ \ \ \ \ \&\ \ \ \ \ \ \mu _{2}\left ( t \right )=\sum_{I}^{i=t+1}\frac{iP\left ( i \right )}{q_{2}\left ( t \right )}$$

$$\sigma _{1}^{2}\left ( t \right )=\sum_{i=1}^{t}\left [ i - \mu _{1} \left ( t \right )\right ]^{2}\frac{P\left ( i \right )}{q_{1}\left ( t \right )}\ \ \ \ \&\ \ \ \ \sigma _{2}^{2}\left ( t \right )=\sum_{i=t+1}^{I}\left [ i - \mu _{1} \left ( t \right )\right ]^{2}\frac{P\left ( i \right )}{q_{2}\left ( t \right )}$$

它实际上找到了一个位于两个峰之间的t值，这样两个类的方差都是最小的。它可以简单地在Python中实现，如下所示：

```python
img = cv.imread('noisy2.png',0)
blur = cv.GaussianBlur(img,(5,5),0)
# find normalized_histogram, and its cumulative distribution function
hist = cv.calcHist([blur],[0],None,[256],[0,256])
hist_norm = hist.ravel()/hist.max()
Q = hist_norm.cumsum()
bins = np.arange(256)
fn_min = np.inf
thresh = -1
for i in xrange(1,256):
p1,p2 = np.hsplit(hist_norm,[i]) # probabilities
q1,q2 = Q[i],Q[255]-Q[i] # cum sum of classes
b1,b2 = np.hsplit(bins,[i]) # weights
# finding means and variances
m1,m2 = np.sum(p1*b1)/q1, np.sum(p2*b2)/q2
v1,v2 = np.sum(((b1-m1)**2)*p1)/q1,np.sum(((b2-m2)**2)*p2)/q2
# calculates the minimization function
fn = v1*q1 + v2*q2
if fn < fn_min:
fn_min = fn
thresh = i
# find otsu's threshold value with OpenCV function
ret, otsu = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
print( "{} {}".format(thresh,ret) )
```

**注意：这里的一些功能可能是之前没有讲过的，但我们将在后面的章节中介绍它们**



















