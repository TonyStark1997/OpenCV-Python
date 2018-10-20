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

## 四、平滑图像

***

### 目标：

本章节您需要学习以下内容:

    *使用各种低通滤波器模糊图像
    *将定制过滤器应用于图像（2D卷积）

### 1、2D卷积（图像过滤）

与一维信号一样，图像也可以使用各种低通滤波器（LPF），高通滤波器（HPF）等进行滤波.LPF有助于消除噪声，模糊图像等.HPF滤波器有助于找到边缘 图片。

OpenCV提供了一个函数cv.filter2D（）来将内核与图像进行卷积。 例如，我们将尝试对图像进行平均滤波。 5x5平均滤波器内核如下所示：

$$K=\frac{1}{25}\begin{bmatrix}
\ 1\ \ 1 \ \ 1 \ \ 1\ \ 1\\ 
\ 1\ \ 1 \ \ 1 \ \ 1\ \ 1\\ 
\ 1\ \ 1 \ \ 1 \ \ 1\ \ 1\\ 
\ 1\ \ 1 \ \ 1 \ \ 1\ \ 1\\ 
\ 1\ \ 1 \ \ 1 \ \ 1\ \ 1 
\end{bmatrix}$$

操作是这样的：将此内核保持在像素上方，添加该内核下方的所有25个像素，取其平均值并用新的平均值替换中心像素。它继续对图像中的所有像素执行此操作。试试这段代码并检查结果：

```python
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread('opencv_logo.png')
kernel = np.ones((5,5),np.float32)/25
dst = cv.filter2D(img,-1,kernel)
plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(dst),plt.title('Averaging')
plt.xticks([]), plt.yticks([])
plt.show()
```

窗口将如下图显示：

![image9](https://raw.githubusercontent.com/TonyStark1997/OpenCV-Python/master/4.Image%20Processing%20in%20OpenCV/Image/image9.png)

### 2、图像模糊（图像平滑）

通过将图像与低通滤波器内核卷积来实现平滑图像。它有助于消除噪音，从图像中去除了高频内容（例如：噪声，边缘）。因此在此操作中边缘会模糊一点。 （有平滑的技术也不会平滑边缘）。OpenCV主要提供四种平滑技术。

#### （1）平均

这是通过将图像与标准化的盒式过滤器进行卷积来完成的。它只取内核区域下所有像素的平均值并替换中心元素。这是由函数cv.blur（）或cv.boxFilter（）完成的。查看文档以获取有关内核的更多详细信息。我们应该指定内核的宽度和高度，3x3标准化的盒式过滤器如下所示：

$$K=\frac{1}{9}\begin{bmatrix}
\ 1 \ \ 1\ \ 1\\ 
\ 1 \ \ 1\ \ 1\\ 
\ 1 \ \ 1\ \ 1 
\end{bmatrix}$$

**注意：如果不想使用规范化的框过滤器，请使用cv.boxFilter（）。将参数normalize = False传递给函数。**

使用5x5大小的内核检查下面的示例演示：

```python
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
img = cv.imread('opencv-logo-white.png')
blur = cv.blur(img,(5,5))
plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blur),plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.show()

```

窗口将如下图显示：

![image10](https://raw.githubusercontent.com/TonyStark1997/OpenCV-Python/master/4.Image%20Processing%20in%20OpenCV/Image/image10.png)

#### （2）高斯模糊

下面使用高斯核代替盒式滤波器。它是通过函数cv.GaussianBlur（）完成的。我们应该指定内核的宽度和高度，它应该是正数并且是奇数。我们还应该分别指定X和Y方向的标准偏差sigmaX和sigmaY。如果仅指定了sigmaX，则sigmaY与sigmaX相同。如果两者都为零，则根据内核大小计算它们。高斯模糊在从图像中去除高斯噪声方面非常有效。

如果需要，可以使用函数cv.getGaussianKernel（）创建高斯内核。

上面的代码可以修改为高斯模糊：

```python
blur = cv.GaussianBlur(img,(5,5),0)
```

窗口将如下图显示：

![image11](https://raw.githubusercontent.com/TonyStark1997/OpenCV-Python/master/4.Image%20Processing%20in%20OpenCV/Image/image11.png)

#### （3）中位数模糊

这里，函数cv.medianBlur（）取内核区域下所有像素的中值，并用该中值替换中心元素。这对图像中的椒盐噪声非常有效。有趣的是，在上述滤波器中，中心元素是新计算的值，其可以是图像中的像素值或新值。但在中值模糊中，中心元素总是被图像中的某个像素值替换,它有效地降低了噪音。其内核大小应为正整数。

在这个演示中，我为原始图像添加了50％的噪点并应用了中值模糊。检查结果：

```python
median = cv.medianBlur(img,5)
```

窗口将如下图显示：

![image12](https://raw.githubusercontent.com/TonyStark1997/OpenCV-Python/master/4.Image%20Processing%20in%20OpenCV/Image/image12.png)

#### （4）双边过滤

cv.bilateralFilter（）在降低噪音方面非常有效，同时保持边缘清晰。但与其他过滤器相比，操作速度较慢。我们已经看到高斯滤波器采用像素周围的邻域并找到其高斯加权平均值。该高斯滤波器仅是空间的函数，即在滤波时考虑附近的像素。它没有考虑像素是否具有几乎相同的强度。它不考虑像素是否是边缘像素。所以它也模糊了边缘，我们不想这样做。

双边滤波器在空间中也采用高斯滤波器，但是还有一个高斯滤波器是像素差的函数。空间的高斯函数确保仅考虑附近的像素用于模糊，而强度差的高斯函数确保仅考虑具有与中心像素相似的强度的像素用于模糊。因此它保留了边缘，因为边缘处的像素将具有较大的强度变化。

下面的示例显示使用双边过滤器（有关参数的详细信息，请访问docs）。

```python
blur = cv.bilateralFilter(img,9,75,75)
```

窗口将如下图显示：

![image13](https://raw.githubusercontent.com/TonyStark1997/OpenCV-Python/master/4.Image%20Processing%20in%20OpenCV/Image/image13.png)

## 五、形态转换

***

### 目标：

本章节您需要学习以下内容:

    *我们将学习不同的形态学操作，如侵蚀，膨胀，开放，关闭等。
    *我们将看到不同的函数，如：cv.erode（），cv.dilate（），cv.morphologyEx（）等。

### 理论

形态变换是基于图像形状的一些简单操作。它通常在二进制图像上执行。它需要两个输入，一个是我们的原始图像，第二个是称为结构元素或内核，它决定了操作的性质。侵蚀和膨胀是两个基本的形态学运算符。然后它的变体形式如Opening，Closing，Gradient等也发挥作用。 我们将在以下图片的帮助下逐一看到它们：

![image14](https://raw.githubusercontent.com/TonyStark1997/OpenCV-Python/master/4.Image%20Processing%20in%20OpenCV/Image/image14.png)

### 1、侵蚀

侵蚀的基本思想就像土壤侵蚀一样，它会侵蚀前景物体的边界（总是试图保持前景为白色）。它有什么作用？内核在图像中滑动（如在2D卷积中），只有当内核下的所有像素都是1时，原始图像中的像素（1或0）才会被认为是1，否则它会被侵蚀（变为零）。

所以侵蚀作用后，边界附近的所有像素都将被丢弃，具体取决于内核的大小。因此，前景对象的厚度或大小减小，或者图像中的白色区域减小。它有助于消除小的白噪声（正如我们在色彩空间章节中看到的那样），分离两个连接的对象等。

在这里，作为一个例子，我将使用一个5x5内核，其中包含完整的内核。 让我们看看它是如何工作的：

```python
import cv2 as cv
import numpy as np
img = cv.imread('j.png',0)
kernel = np.ones((5,5),np.uint8)
erosion = cv.erode(img,kernel,iterations = 1)
```
窗口将如下图显示：

![image15](https://raw.githubusercontent.com/TonyStark1997/OpenCV-Python/master/4.Image%20Processing%20in%20OpenCV/Image/image15.png)

### 2、扩张

它恰好与侵蚀相反。这里，如果内核下的至少一个像素为“1”，则像素元素为“1”。因此它增加了图像中的白色区域或前景对象的大小增加。通常，在去除噪音的情况下，侵蚀之后是扩张。因为，侵蚀会消除白噪声，但它也会缩小我们的物体,所以我们扩大它。由于噪音消失了，它们不会再回来，但我们的物体区域会增加。它也可用于连接对象的破碎部分。

```python
dilation = cv.dilate(img,kernel,iterations = 1)
```

窗口将如下图显示：

![image16](https://raw.githubusercontent.com/TonyStark1997/OpenCV-Python/master/4.Image%20Processing%20in%20OpenCV/Image/image16.png)

### 3、开放

开放只是侵蚀之后紧接着做扩张处理的合成步骤。如上所述，它有助于消除噪音。这里我们使用函数cv.morphologyEx（）

```python
opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
```

窗口将如下图显示：

![image17](https://raw.githubusercontent.com/TonyStark1997/OpenCV-Python/master/4.Image%20Processing%20in%20OpenCV/Image/image17.png)

### 4、关闭

关闭与开放，扩张和侵蚀相反。它可用于过滤前景对象内的小孔或对象上的小黑点。

```python
closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
```

窗口将如下图显示：

![image18](https://raw.githubusercontent.com/TonyStark1997/OpenCV-Python/master/4.Image%20Processing%20in%20OpenCV/Image/image18.png)

### 5、形态学梯度

它的处理结果是显示膨胀和侵蚀之间的差异。

结果看起来像对象的轮廓。

```python
gradient = cv.morphologyEx(img, cv.MORPH_GRADIENT, kernel)
```

窗口将如下图显示：

![image19](https://raw.githubusercontent.com/TonyStark1997/OpenCV-Python/master/4.Image%20Processing%20in%20OpenCV/Image/image19.png)

### 6、大礼帽

它的处理结果是输入图像和图像打开之间的区别。下面的示例是针对9x9内核完成的。

```python
tophat = cv.morphologyEx(img, cv.MORPH_TOPHAT, kernel)
```

窗口将如下图显示：

![image20](https://raw.githubusercontent.com/TonyStark1997/OpenCV-Python/master/4.Image%20Processing%20in%20OpenCV/Image/image20.png)

### 7、黑帽子

它是输入图像关闭和输入图像之间的差异。

```python
blackhat = cv.morphologyEx(img, cv.MORPH_BLACKHAT, kernel)
```

窗口将如下图显示：

![image21](https://raw.githubusercontent.com/TonyStark1997/OpenCV-Python/master/4.Image%20Processing%20in%20OpenCV/Image/image21.png)

### 8、结构元素

我们在Numpy的帮助下手动创建了前面示例中的结构元素。它是矩形，但在某些情况下可能需要椭圆或圆形内核。所以为此，OpenCV有一个函数cv.getStructuringElement（）。只需传递内核的形状和大小，即可获得所需的内核。

```python
# Rectangular Kernel
>>> cv.getStructuringElement(cv.MORPH_RECT,(5,5))
array([[1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1]], dtype=uint8)
# Elliptical Kernel
>>> cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
array([[0, 0, 1, 0, 0],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1],
       [0, 0, 1, 0, 0]], dtype=uint8)
# Cross-shaped Kernel
>>> cv.getStructuringElement(cv.MORPH_CROSS,(5,5))
array([[0, 0, 1, 0, 0],
       [0, 0, 1, 0, 0],
       [1, 1, 1, 1, 1],
       [0, 0, 1, 0, 0],
       [0, 0, 1, 0, 0]], dtype=uint8)
```

## 六、图像渐变

***

### 目标：

本章节您需要学习以下内容:

    *查找图像渐变，边缘等
    *我们将看到以下函数：cv.Sobel（），cv.Scharr（），cv.Laplacian（）等

### 1、理论

OpenCV提供三种类型的梯度滤波器或高通滤波器，Sobel，Scharr和Laplacian。 我们将看到他们中的每一个。

#### （1）Sobel和Scharr衍生物

Sobel算子是高斯联合平滑加微分运算，因此它更能抵抗噪声。你可以指定要采用的导数的方向，垂直或水平（分别通过参数，yorder和xorder），你还可以通过参数ksize指定内核的大小。如果ksize = -1，则使用3x3 Scharr滤波器，其结果优于3x3 Sobel滤波器。请参阅所用内核的文档。

#### （2）拉普拉斯衍生物

它计算由关系给出的图像的拉普拉斯算子，$\Delta src= \frac{\partial ^{2}src}{\partial x^{2}}+ \frac{\partial ^{2}src}{\partial y^{2}}$，其中使用Sobel导数找到每个导数。 如果ksize = 1，则使用以下内核进行过滤：

$$kernel=\begin{bmatrix}
\ 0\ \ \ \ 1\ \ \ \ 0\\ 
\ 1\ -4\ \ 1\\ 
\ 0\ \ \ \ 1\ \ \ \ 0 
\end{bmatrix}$$

### 2、代码实现

下面的代码显示了单个图表中的所有运算符，所有内核都是5x5大小。输出图像的深度为-1，以获得np.uint8类型的结果。

```python
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread('dave.jpg',0)
laplacian = cv.Laplacian(img,cv.CV_64F)
sobelx = cv.Sobel(img,cv.CV_64F,1,0,ksize=5)
sobely = cv.Sobel(img,cv.CV_64F,0,1,ksize=5)
plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
plt.show()
```

窗口将如下图显示：

![image22](https://raw.githubusercontent.com/TonyStark1997/OpenCV-Python/master/4.Image%20Processing%20in%20OpenCV/Image/image22.png)

### 3、一个重要的事情

在我们的上一个示例中，输出数据类型为cv.CV_8U或np.uint8，但是这有一个小问题，将黑到白转换视为正斜率（它具有正值），而将白到黑转换视为负斜率（它具有负值）。因此，当你将数据转换为np.uint8时，所有负斜率都为零。简单来说，你错过了这个优势。

如果要检测两个边，更好的选择是将输出数据类型保持为某些更高的形式，如cv.CV_16S，cv.CV_64F等，取其绝对值，然后转换回cv.CV_8U。下面的代码演示了水平Sobel滤波器的这个过程以及结果的差异。

```python
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread('box.png',0)
# Output dtype = cv.CV_8U
sobelx8u = cv.Sobel(img,cv.CV_8U,1,0,ksize=5)
# Output dtype = cv.CV_64F. Then take its absolute and convert to cv.CV_8U
sobelx64f = cv.Sobel(img,cv.CV_64F,1,0,ksize=5)
abs_sobel64f = np.absolute(sobelx64f)
sobel_8u = np.uint8(abs_sobel64f)
plt.subplot(1,3,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,2),plt.imshow(sobelx8u,cmap = 'gray')
plt.title('Sobel CV_8U'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,3),plt.imshow(sobel_8u,cmap = 'gray')
plt.title('Sobel abs(CV_64F)'), plt.xticks([]), plt.yticks([])
plt.show()
```

窗口将如下图显示：

![image23](https://raw.githubusercontent.com/TonyStark1997/OpenCV-Python/master/4.Image%20Processing%20in%20OpenCV/Image/image23.png)

## 七、Canny边缘检测

***

### 目标：

本章节您需要学习以下内容:

    *Canny边缘检测的概念
    *OpenCV的功能：cv.Canny（）

### 1、理论

Canny边缘检测是一种流行的边缘检测算法，它是由John F. Canny开发的

这是一个多阶段算法，我们将了解其中的每个阶段。

#### （1）降噪

由于边缘检测易受图像中的噪声影响，因此第一步是使用5x5高斯滤波器去除图像中的噪声。我们在之前的章节中已经看到了这一点。

#### （2）寻找图像的强度梯度

然后在水平和垂直方向上用Sobel核对平滑后的图像进行滤波，以获得水平方向（$G_{x}$）和垂直方向（$G_{y}$）的一阶导数。从这两个图像中，我们可以找到每个像素的边缘梯度和方向，如下所示：

$$Edge\_Gradient\left ( G \right )= \sqrt{G_{x}^{2}+G_{y}^{2}}$$

$$Angle\left ( \theta  \right )= tan^{-1}\left ( \frac{G_{y}}{G_{x}} \right )$$

渐变方向始终垂直于边缘。它被四舍五入到表示垂直，水平和两个对角线方向的四个角度中的一个。

#### （3）非最大抑制

在获得梯度幅度和方向之后，完成图像的全扫描以去除可能不构成边缘的任何不需要的像素。为此，在每个像素处，检查像素是否是其在梯度方向上的邻域中的局部最大值。检查下图：

![image24](https://raw.githubusercontent.com/TonyStark1997/OpenCV-Python/master/4.Image%20Processing%20in%20OpenCV/Image/image24.png)

A点位于边缘（垂直方向）。渐变方向与边缘垂直。 B点和C点处于梯度方向。因此，用点B和C检查点A，看它是否形成局部最大值。如果是这样，则考虑下一阶段，否则，它被抑制（置零）。

简而言之，您得到的结果是具有“细边”的二进制图像。

#### （4）滞后阈值

这个阶段决定哪些边缘都是边缘，哪些边缘不是边缘。为此，我们需要两个阈值，minVal和maxVal。强度梯度大于maxVal的任何边缘肯定是边缘，而minVal以下的边缘肯定是非边缘的，因此被丢弃。位于这两个阈值之间的人是基于其连通性的分类边缘或非边缘。如果它们连接到“可靠边缘”像素，则它们被视为边缘的一部分。否则，他们也被丢弃。见下图：

![image25](https://raw.githubusercontent.com/TonyStark1997/OpenCV-Python/master/4.Image%20Processing%20in%20OpenCV/Image/image25.png)

边缘A高于maxVal，因此被视为“确定边缘”。虽然边C低于maxVal，但它连接到边A，因此也被视为有效边，我们得到完整的曲线。但边缘B虽然高于minVal并且与边缘C的区域相同，但它没有连接到任何“可靠边缘”，因此被丢弃。因此，我们必须相应地选择minVal和maxVal才能获得正确的结果。

假设边是长线，这个阶段也会消除小像素噪声。

所以我们最终得到的是图像中的强边缘。

### 2、OpenCV中的Canny边缘检测

OpenCV将以上所有内容放在单个函数cv.Canny（）中。我们将看到如何使用它。第一个参数是我们的输入图像。第二个和第三个参数分别是我们的minVal和maxVal。第三个参数是aperture_size。它是用于查找图像渐变的Sobel内核的大小。默认情况下，它是3.最后一个参数是L2gradient，它指定用于查找梯度幅度的等式。如果它是True，它使用上面提到的更准确的等式，否则它使用这个函数：$Edge\_Gradient\left ( G \right )= \left | G_{x} \right |+\left | G_{y} \right |$。默认情况下，它为False。

```python
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread('messi5.jpg',0)
edges = cv.Canny(img,100,200)
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()
```

窗口将如下图显示：

![image26](https://raw.githubusercontent.com/TonyStark1997/OpenCV-Python/master/4.Image%20Processing%20in%20OpenCV/Image/image26.png)






























