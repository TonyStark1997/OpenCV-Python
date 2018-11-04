# 第五章：特征提取与描述

本章节你将学习图像的主要特征、Harris角点检测、Shi-Tomasi角点检测、SIFT、SURF、特征匹配等OpenCV图像特征提取与描述的相关内容。

更多内容请关注我的GitHub库：https://github.com/TonyStark1997，如果喜欢，star并follow我！

***

## 一、理解图像特征

***

### 目标：

本章节你需要学习以下内容:

    *在本章中，我们将尝试了解哪些是图像的特征，理解为什么图像特征很重要，理解为什么角点很重要等等。

### 解释

相信大多数人都玩过拼图游戏。你会得到许多零零散散的碎片，然后需要正确地组装它们以形成一个大的完整的图像。问题是，你是怎么做到的？如何将相同的理论应用到计算机程序中，以便计算机可以玩拼图游戏？如果计算机可以玩拼图游戏，为什么我们不能给计算机提供很多真实自然景观的真实图像，并告诉它将所有这些图像拼接成一个大的单个图像？如果计算机可以将几个零散图像拼接成一个，那么如何提供大量建筑物或任何结构的图片并告诉计算机从中创建3D模型呢？

问题和想象力可以是无边无际的，但这一切都取决于最基本的问题：你是如何玩拼图游戏的？你如何将大量的混乱图像片段排列成一个大的完整的图像？如何将大量零散图像拼接成整体图像？

答案是，我们正在寻找独特的特定模式或特定功能，可以轻松跟踪并轻松比较。如果我们找到这样一个特征的定义，我们可能会发现很难用文字表达，但我们知道它们是什么。如果有人要求你指出可以在多个图像之间进行比较的一个好的功能，你可以指出一个。这就是为什么即使是小孩子也可以简单地玩这些游戏。我们在图像中搜索这些特征，找到它们，在其他图像中查找相同的特征并拼凑它们。（在拼图游戏中，我们更多地关注不同图像的连续性）。所有这些能力都是我们天生所具备的。

因此，我们的一个基本问题扩展到更多，但变得更具体。这些功能是什么？（答案对于计算机也应该是可以理解的。）

很难说人类如何找到这些特征。这已经在我们的大脑中编程。但是如果我们深入研究一些图片并搜索不同的图案，我们会发现一些有趣的东西。例如，拍下图片：

![image1](https://raw.githubusercontent.com/TonyStark1997/OpenCV-Python/master/5.Feature%20Detection%20and%20Description/Image/image1.jpg)

图像非常简单。在图像的顶部，给出了六个小图像补丁。你的问题是在原始图像中找到这些补丁的确切位置。你能找到多少正确的结果？

A和B是平坦的表面，它们分布在很多区域。很难找到这些补丁的确切位置。

C和D要简单得多。它们是建筑物的边缘。你可以找到一个大概的位置，但确切的位置仍然很困难。这是因为沿着边缘的模式是相同的。然而，在边缘，它是不同的。因此，与平坦区域相比，边缘是更好的特征，但是不够好（用于比较边缘的连续性在拼图中是好的）。

最后，E和F是建筑物的一些角点。它们很容易找到。因为在角点，无论你移动这个补丁，它都会有所不同。所以它们可以被认为是很好的功能。所以现在我们进入更简单（和广泛使用的图像）以便更好地理解。

![image2](https://raw.githubusercontent.com/TonyStark1997/OpenCV-Python/master/5.Feature%20Detection%20and%20Description/Image/image2.jpg)

就像上面一样，蓝色斑块是平坦的区域，很难找到和跟踪。无论你移动蓝色补丁，它看起来都一样。黑色贴片有边缘。如果沿垂直方向（即沿着渐变方向）移动它会改变。沿边缘移动（平行于边缘），看起来一样。对于红色补丁，它是一个角点。无论你移动补丁，它看起来都不同，意味着它是独一无二的。所以基本上，角点被认为是图像中的好特征。（不仅仅是角点，在某些情况下，blob被认为是很好的特征）。

所以现在我们回答了我们的问题，“这些功能是什么？”。但接下来的问题就出现了。我们如何找到它们？或者我们如何找到角点？我们以直观的方式回答了这一点，即在图像中寻找在其周围的所有区域中移动（少量）时具有最大变化的区域。在接下来的章节中，这将被投射到计算机语言中。因此，查找这些图像特征称为特征检测。

我们在图像中找到了这些功能。一旦找到它，你应该能够在其他图像中找到相同的内容。这是怎么做到的？我们用一个区域围绕这个特征，我们用自己的话解释它，比如“上部是蓝天，下部是建筑物的区域，那个建筑物上有玻璃等”，你在另一个地方寻找相同的区域图片。基本上，你正在描述该功能。类似地，计算机还应该描述特征周围的区域，以便它可以在其他图像中找到它。所谓的描述称为特征描述。获得这些功能及其描述后，你可以在所有图像中找到相同的功能并对齐它们，将它们拼接在一起或做任何你想做的事情。

因此，在本单元中，我们正在寻找OpenCV中的不同算法来查找功能，描述功能，匹配它们等。

## 二、Harris角点检测

***

### 目标：

本章节你需要学习以下内容:

    *我们将了解Harris Corner Detection背后的概念。
    *我们将看到函数：cv.cornerHarris()，cv.cornerSubPix()
    
### 1、理论

在上一节我们已经知道了角点的一个特性：向任何方向移动变化都很大。Chris_Harris 和 Mike_Stephens 早在 1988 年的文章《A CombinedCorner and Edge Detector》中就已经提出了焦点检测的方法，被称为Harris 角点检测。他把这个简单的想法转换成了数学形式。将窗口向各个方向移动（u，v）然后计算所有差异的总和。表示如下：

$$E(u,v) = \sum_{x,y} \underbrace{w(x,y)}_\text{window function} \, [\underbrace{I(x+u,y+v)}_\text{shifted intensity}-\underbrace{I(x,y)}_\text{intensity}]^2$$

窗口函数可以是正常的矩形窗口也可以是对每一个像素给予不同权重的高斯窗口。

角点检测中要使 E (µ,ν) 的值最大。这就是说必须使方程右侧的第二项的取值最大。对上面的等式进行泰勒级数展开然后再通过几步数学换算（可以参考其他标准教材），我们得到下面的等式：

$$E(u,v) \approx \begin{bmatrix} u & v \end{bmatrix} M \begin{bmatrix} u \\ v \end{bmatrix}$$

其中：

$$M = \sum_{x,y} w(x,y) \begin{bmatrix}I_x I_x & I_x I_y \\ I_x I_y & I_y I_y \end{bmatrix}$$

这里，$I_x$和$I_y$分别是x和y方向上的图像导数。（可以使用函数cv.Sobel()轻松找到）。

然后是主要部分。在此之后，他们创建了一个分数，基本上是一个等式，它将确定一个窗口是否可以包含一个角点。

$$R = det(M) - k(trace(M))^2$$

其中：

* $R = det(M) - k(trace(M))^2$
* $trace(M) = \lambda_1 + \lambda_2$
* $\lambda_1$和$\lambda_2$是M的本征值

因此，这些特征值的值决定区域是角点，边缘还是平坦。

* 当$|R|$很小，当$\lambda_1$和$\lambda_2$很小时，该区域是平坦的。
* 当$R<0$时，在$\lambda_1 >> \lambda_2$时发生，反之亦然，该区域是边缘。
* 当R很大时，当$\lambda_1$和$\lambda_2$大并且$\lambda_1 \sim \lambda_2$时发生，该区域是拐角。
* 
它可以用下面这张很好理解的图片表示：

![image3](https://raw.githubusercontent.com/TonyStark1997/OpenCV-Python/master/5.Feature%20Detection%20and%20Description/Image/image3.jpg)

因此Harris角点检测的结果是一个由角点分数构成的灰度图像。选取适当的阈值对结果图像进行二值化我们就检测到了图像中的角点。我们将用一个简单的图片来演示一下。

### 2、OpenCV中的Harris角点探测器

为此，OpenCV具有函数cv.cornerHarris()。它的参数是：

* img - 输入图像，应该是灰度和float32类型。
* blockSize - 考虑角点检测的邻域大小
* ksize - 使用的Sobel衍生物的孔径参数。
* k - 方程中的Harris检测器自由参数。
* 
请参阅以下示例：

```python
import numpy as np
import cv2 as cv

filename = 'chessboard.png'
img = cv.imread(filename)
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

gray = np.float32(gray)
dst = cv.cornerHarris(gray,2,3,0.04)

#result is dilated for marking the corners, not important
dst = cv.dilate(dst,None)

# Threshold for an optimal value, it may vary depending on the image.
img[dst>0.01*dst.max()]=[0,0,255]

cv.imshow('dst',img)
if cv.waitKey(0) & 0xff == 27:
    cv.destroyAllWindows()
```

以下是三个结果：

![image4](https://raw.githubusercontent.com/TonyStark1997/OpenCV-Python/master/5.Feature%20Detection%20and%20Description/Image/image4.jpg)

### 3、具有亚像素精度的角点

有时，你可能需要以最高精度找到角点。OpenCV附带了一个函数cv.cornerSubPix()，它进一步细化了以亚像素精度检测到的角点。以下是一个例子。像往常一样，我们需要先找到Harris的角点。然后我们传递这些角的质心（角点处可能有一堆像素，我们采用它们的质心）来细化它们。Harris角以红色像素标记，精致角以绿色像素标记。对于此函数，我们必须定义何时停止迭代的标准。我们在指定的迭代次数或达到一定精度后停止它，以先发生者为准。我们还需要定义它将搜索角点的邻域大小。

```python
import numpy as np
import cv2 as cv

filename = 'chessboard2.jpg'
img = cv.imread(filename)
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

# find Harris corners
gray = np.float32(gray)
dst = cv.cornerHarris(gray,2,3,0.04)
dst = cv.dilate(dst,None)
ret, dst = cv.threshold(dst,0.01*dst.max(),255,0)
dst = np.uint8(dst)

# find centroids
ret, labels, stats, centroids = cv.connectedComponentsWithStats(dst)

# define the criteria to stop and refine the corners
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
corners = cv.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)

# Now draw them
res = np.hstack((centroids,corners))
res = np.int0(res)
img[res[:,1],res[:,0]]=[0,0,255]
img[res[:,3],res[:,2]] = [0,255,0]

cv.imwrite('subpixel5.png',img)
```

下面是结果，其中一些重要位置显示在缩放窗口中以显示：

![image5](https://raw.githubusercontent.com/TonyStark1997/OpenCV-Python/master/5.Feature%20Detection%20and%20Description/Image/image5.jpg)