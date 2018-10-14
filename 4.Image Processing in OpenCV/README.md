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

![image1]()

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

![image2]()

### 3.旋转

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

![image3]()

### 4.仿射变换

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

![image4]()

### 5.透视转型

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

![image5]()
