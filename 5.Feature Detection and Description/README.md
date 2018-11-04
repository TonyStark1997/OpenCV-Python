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

## 三、Shi-Tomasi角点探测器和适合于跟踪的图像特征

***

### 目标：

本章节你需要学习以下内容:

    *我们将了解另一个角点探测器：Shi-Tomasi角点探测器
    *我们将看到函数：cv.goodFeaturesToTrack()
    
### 1、理论

在上一小节，我们看到了Harris角点检测。1994年晚些时候，J.Shi和C.Tomasi在他们的论文《Good Features to Track》中做了一个小修改，与Harris角点检测相比显示出更好的结果。Harris角点探测器的评分功能由下式给出：

$$R = \lambda_1 \lambda_2 - k(\lambda_1+\lambda_2)^2$$

除此之外，Shi-Tomasi提出：

$$R = min(\lambda_1, \lambda_2)$$

如果它大于阈值，则将其视为拐角。如果我们像在Harris角点检测器中那样在$\lambda_1 - \lambda_2$空间中绘制它，我们得到如下图像：

![image6](https://raw.githubusercontent.com/TonyStark1997/OpenCV-Python/master/5.Feature%20Detection%20and%20Description/Image/image6.jpg)

从图中可以看出，只有当$\lambda_1$和$\lambda_2$高于最小值λmin时，它才被视为一个角（绿色区域）。

### 2、代码实现

OpenCV有一个函数cv.goodFeaturesToTrack()。 它通过Shi-Tomasi方法（或Harris角点检测，如果你指定它）在图像中找到N个最强角。像往常一样，图像应该是灰度图像。然后指定要查找的角点数。然后指定质量等级，该等级是0-1之间的值，表示低于每个人被拒绝的角点的最低质量。然后我们提供检测到的角之间的最小欧氏距离。

利用所有这些信息，该函数可以在图像中找到角点。低于质量水平的所有角点都被拒绝。然后它根据质量按降序对剩余的角进行排序。然后功能占据第一个最强的角点，抛弃最小距离范围内的所有角点并返回N个最强的角点。

在下面的示例中，我们将尝试找到25个最佳角点：

```python
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('blox.jpg')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

corners = cv.goodFeaturesToTrack(gray,25,0.01,10)
corners = np.int0(corners)

for i in corners:
    x,y = i.ravel()
    cv.circle(img,(x,y),3,255,-1)
    
plt.imshow(img),plt.show()
```

结果如下图所示：

![image7](https://raw.githubusercontent.com/TonyStark1997/OpenCV-Python/master/5.Feature%20Detection%20and%20Description/Image/image7.jpg)

我们以后会发现这个函数很适合在目标跟踪中使用。

## 四、介绍SIFT(Scale-Invariant Feature Trans-form)

***

### 目标：

本章节你需要学习以下内容:

    *我们将了解SIFT算法的概念
    *我们将学习如何找到SIFT关键点和描述符。
    
### 1、理论
    
在上一小节中，我们看到了一些角点探测器，如Harris角点探测器等。它们具有旋转不变的特性，这意味着，即使图像旋转，我们也可以找到相同的角点。很明显，因为角落在旋转的图像中也是角点。但是缩放呢？如果缩放图像，则角点可能不是角点。例如，检查下面的简单图像。当在同一窗口中放大时，小窗口内的小图像中的角是平坦的。所以Harris的角点不是规模不变的。

![image8](https://raw.githubusercontent.com/TonyStark1997/OpenCV-Python/master/5.Feature%20Detection%20and%20Description/Image/image8.jpg)

因此，2004年，不列颠哥伦比亚大学D.Lowe在他的论文中提出了一种新的算法，尺度不变特征变换（SIFT），从尺度不变关键点的独特图像特征，提取关键点并计算其描述符。（本文易于理解并被认为是SIFT上最好的材料。所以这个解释只是本文的简短摘要）。

SIFT算法主要涉及四个步骤。我们将逐一看到它们。

#### （1）尺度空间极值检测

从上图我们可以很明显的看出来在不同的尺度空间不能使用相同的窗口检测极值点。对小的角点要用小的窗口，对大的角点只能使用大的窗口。为了达到这个目的我们要使用尺度空间滤波器。（尺度空间滤波器可以使用一些列具有不同方差$\sigma$的高斯卷积核构成）。使用具有不同方差值$\sigma$的高斯拉普拉斯算子（LoG）对图像进行卷积，LoG 由于具有不同的方差值$\sigma$所以可以用来检测不同大小的斑点（当 LoG 的方差$\sigma$与斑点直径相等时能够使斑点完全平滑）。简单来说方差$\sigma$就是一个尺度变换因子。例如，上图中使用一个小方差$\sigma$的高斯卷积核是可以很好的检测出小的角点，而使用大方差$\sigma$的高斯卷积核时可以很好的检测除大的角点。所以我们可以在尺度空间和二维平面中检测到局部最大值，如$(x,y,\sigma)$, 这表示在$\sigma$尺度中$(x,y)$点可能是一个关键点。（高斯方差的大小与窗口的大小存在一个倍数关系：窗口大小等于 6 倍方差加 1，所以方差的大小也决定了窗口大小）

但是这个 LoG 的计算量非常大，所以 SIFT 算法使用高斯差分算子（DoG）来对 LoG 做近似。这里需要再解释一下图像金字塔，我们可以通过减少采样（如只取奇数行或奇数列）来构成一组图像尺寸（1，0.5，0.25 等）不同的金字塔，然后对这一组图像中的每一张图像使用具有不同方差$\sigma$的高斯卷积核构建出具有不同分辨率的图像金字塔（不同的尺度空间）。DoG 就是这组具有不同分辨率的图像金字塔中相邻的两层之间的差值。如下图所示：

![image9](https://raw.githubusercontent.com/TonyStark1997/OpenCV-Python/master/5.Feature%20Detection%20and%20Description/Image/image9.jpg)

在 DoG 搞定之后，就可以在不同的尺度空间和 2D 平面中搜索局部最大值了。对于图像中的一个像素点而言，它需要与自己周围的 8 邻域，以及尺度空间中上下两层中的相邻的 18（2x9）个点相比。如果是局部最大值，它就可能是一个关键点。基本上来说关键点是图像在相应尺度空间中的最好代表。如下图所示：

![image10](https://raw.githubusercontent.com/TonyStark1997/OpenCV-Python/master/5.Feature%20Detection%20and%20Description/Image/image10.jpg)

该算法的作者在文章中给出了 SIFT 参数的经验值：octaves=4（通过降低采样从而减小图像尺寸，构成尺寸减小的图像金字塔，尺度空间为 5，也就是每个尺寸使用 5 个不同方差的高斯核进行卷积，初始方差是 1.6，$\sigma=1.6$,$k=\sqrt{2}$等作为最优值。

#### （2）关键点（极值点）定位

一旦找到关键点，我们就要对它们进行修正从而得到更准确的结果。可以使用尺度空间的泰勒级数展开来获得极值的准确位置，如果极值点的灰度值小于阈值（0.03）就会被忽略掉。在 OpenCV 中这种阈值被称为contrastThreshold。

DoG 算法对边界非常敏感，所以我们必须要把边界去除。前面我们讲的Harris 算法除了可以用于角点检测之外还可以用于检测边界。作者就是使用了同样的思路。作者使用 2x2 的 Hessian 矩阵计算主曲率。从 Harris 角点检测的算法中，我们知道当一个特征值远远大于另外一个特征值时检测到的是边界。

所以他们使用了一个简单的函数，如果比例高于阈值（OpenCV 中称为边界阈值），这个关键点就会被忽略。文章中给出的边界阈值为 10。

所以低对比度的关键点和边界关键点都会被去除掉，剩下的就是我们感兴趣的关键点了。

#### （3）为关键点（极值点）指定方向参数

现在我们要为每一个关键点赋予一个反向参数，这样它才会具有旋转不变性。获取关键点（所在尺度空间）的邻域，然后计算这个区域的梯度级和方向。根据计算得到的结果创建一个含有 36 个 bins（每 10 度一个 bin）的方向直方图。（使用当前尺度空间$\sigma$值的 1.5 倍为方差的圆形高斯窗口和梯度级做权重）。直方图中的峰值为主方向参数，如果其他的任何柱子的高度高于峰值的80% 被认为是辅方向。这就会在相同的尺度空间相同的位置构建除具有不同方向的关键点。这对于匹配的稳定性会有所帮助。

#### （4）关键点描述符

新的关键点描述符被创建了。选取与关键点周围一个 16x16 的邻域，把它分成 16 个 4x4 的小方块，为每个小方块创建一个具有 8 个 bin 的方向直方图。总共加起来有 128 个 bin。由此组成长为 128 的向量就构成了关键点描述符。除此之外还要进行几个测量以达到对光照变化，旋转等的稳定性。

#### （5）关键点匹配

下一步就可以采用关键点特征向量的欧式距离来作为两幅图像中关键点的相似性判定度量。取第一个图的某个关键点，通过遍历找到第二幅图像中的距离最近的那个关键点。但有些情况下，第二个距离最近的关键点与第一个距离最近的关键点靠的太近。这可能是由于噪声等引起的。此时要计算最近距离与第二近距离的比值。如果比值大于 0.8，就忽略掉。这会去除 90% 的错误匹配，同时只去除 5% 的正确匹配。如文章所说。
这就是 SIFT 算法的摘要。非常推荐你阅读原始文献，这会加深你对算法的理解。请记住这个算法是受专利保护的。所以这个算法包含在 OpenCV 中的收费模块中。

### 2、OpenCV中的SIFT

现在让我们来看看 OpenCV 中关于 SIFT 的函数吧。让我们从关键点检测和绘制开始吧。首先我们要创建对象。我们可以使用不同的参数，这并不是必须的，关于参数的解释可以查看文档。

```python
import numpy as np
import cv2 as cv

img = cv.imread('home.jpg')
gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)

sift = cv.xfeatures2d.SIFT_create()
kp = sift.detect(gray,None)

img=cv.drawKeypoints(gray,kp,img)

cv.imwrite('sift_keypoints.jpg',img)
```

函数 sift.detect() 可以在图像中找到关键点。如果你只想在图像中的一个区域搜索的话，也可以创建一个掩模图像作为参数使用。返回的关键点是一个带有很多不同属性的特殊结构体，这些属性中包含它的坐标(x，y)，有意义的邻域大小，确定其方向的角度、指定关键点强度的响应等。

OpenCV也提供了绘制关键点的函数：cv2.drawKeyPoints()，它可以在关键点的部位绘制一个小圆圈。如果你设置参数为cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS，就会绘制代表关键点大小的圆圈甚至可以绘制除关键点的方向。见下面的例子。

```python
img=cv.drawKeypoints(gray,kp,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv.imwrite('sift_keypoints.jpg',img)
```

结果如下图所示：

![image11](https://raw.githubusercontent.com/TonyStark1997/OpenCV-Python/master/5.Feature%20Detection%20and%20Description/Image/image11.jpg)

现在要计算描述符，OpenCV提供了两种方法。

1. 由于你已经找到了关键点，因此可以调用sift.compute()来计算我们找到的关键点的描述符。例如：kp，des = sift.compute（灰色，kp）
2. 如果你没有找到关键点，请使用函数sift.detectAndCompute()在一个步骤中直接查找关键点和描述符。

我们将看到第二种方法：

```python
sift = cv.xfeatures2d.SIFT_create()
kp, des = sift.detectAndCompute(gray,None)
```

这里kp是关键点列表，des是形状为$Number\_of\_Keypoints \times 128$的numpy数组。

所以我们得到了关键点，描述符等。现在我们想看看如何匹配不同图像中的关键点。 我们将在接下来的章节中学习。

## 五、介绍SURF(Speeded-Up Robust Features)

***

### 目标：

本章节你需要学习以下内容:

    *我们将看到SURF的基础知识
    *我们将在OpenCV中看到SURF功能
    
### 1、理论

在上一节中我们学习了使用 SIFT 算法进行关键点检测和描述。但是这种算法的执行速度比较慢，人们需要速度更快的算法。在 2006 年Bay,H.,Tuytelaars,T. 和 Van Gool,L 共同提出了 SURF（加速稳健特征）算法。跟它的名字一样，这是个算法是加速版的 SIFT。

在 SIFT 中，Lowe 在构建尺度空间时使用 DoG 对 LoG 进行近似。SURF则更进一步，使用盒子滤波器（box_filter）对 LoG 进行近似。下图显示了这种近似。在进行卷积计算时可以利用积分图 像（积分图像的一大特点是：计算图像中某个窗口内所有像素和时，计算量的大小与窗口大小无关），是盒子滤波器的一大优点。而且这种计算可以在不同尺度空间同时进行。同样 SURF 算法计算关键点的尺度和位置是也是依赖与 Hessian 矩阵行列式的。

![image12](https://raw.githubusercontent.com/TonyStark1997/OpenCV-Python/master/5.Feature%20Detection%20and%20Description/Image/image12.jpg)

为了保证特征矢量具有选装不变形，需要对于每一个特征点分配一个主要方向。需要以特征点为中心，以 6s（s 为特征点的尺度）为半径的圆形区域内，对图像进行 Harr 小波相应运算。这样做实际就是对图像进行梯度运算，但是利用积分图像，可以提高计算图像梯度的效率，为了求取主方向值，需哟啊设计一个以方向为中心，张角为 60 度的扇形滑动窗口，以步长为 0.2 弧度左右旋转这个滑动窗口，并对窗口内的图像 Haar 小波的响应值进行累加。主方向为最大的 Haar 响应累加值对应的方向。在很多应用中根本就不需要旋转不变性，所以没有必要确定它们的方向，如果不计算方向的话，又可以使算法提速。SURF 提供了成为 U-SURF 的功能，它具有更快的速度，同时保持了对$\pm 15^{\circ}$旋转的稳定性。OpenCV 对这两种模式同样支持，只需要对参数upright 进行设置，当 upright 为 0 时计算方向，为 1 时不计算方向，同时速度更快。

![image13](https://raw.githubusercontent.com/TonyStark1997/OpenCV-Python/master/5.Feature%20Detection%20and%20Description/Image/image13.jpg)

生成特征点的特征矢量需要计算图像的 Haar 小波响应。在一个矩形的区域内，以特征点为中心，沿主方向将 20s * 20s 的图像划分成 4 * 4 个子块，每个子块利用尺寸 2s 的 Haar 小波模版进行响应计算，然后对响应值进行统计，组成向量$v=( \sum{d_x}, \sum{d_y}, \sum{|d_x|}, \sum{|d_y|})$。这个描述符的长度为 64。降低的维度可以加速计算和匹配，但又能提供更容易区分的特征。

为了增加特征点的独特性，SURF 还提供了一个加强版 128 维的特征描述符。当$d_y>0$和$d_y<0$时分别对$d_x$和$|d_x|$的和进行计算，计算$d_y$和$|d_x|$时也进行区分，这样获得特征就会加倍，但又不会增加计算的复杂度。OpenCV 同样提供了这种功能，当参数 extended 设置为 1 时为 128 维，当参数为 0 时为 64 维，默认情况为 128 维。

另一个重要的改进是使用拉普拉斯算子（Hessian矩阵的迹线）作为潜在兴趣点。它不会增加计算成本，因为它已经在检测期间计算出来。拉普拉斯的标志将黑暗背景上的明亮斑点与相反情况区分开来。在匹配阶段，我们只比较具有相同类型对比度的特征（如下图所示）。这种最小的信息允许更快的匹配，而不会降低描述符的性能。

![image14](https://raw.githubusercontent.com/TonyStark1997/OpenCV-Python/master/5.Feature%20Detection%20and%20Description/Image/image14.jpg)

简单来说 SURF 算法采用了很多方法来对每一步进行优化从而提高速度。分析显示在结果效果相当的情况下 SURF 的速度是 SIFT 的 3 倍。SURF 善于处理具有模糊和旋转的图像，但是不善于处理视角变化和关照变化。

### 2、OpenCV中的SURF

OpenCV就像SIFT一样提供SURF功能。首先使用一些可选条件（如64/128-dim描述符，Upright/Normal SURF等）初始化一个SURF对象。所有详细信息都在文档中进行了详细说明。然后就像我们在SIFT中所做的那样，我们可以使用SURF.detect()，SURF.compute()等来查找关键点和描述符。

首先，我们将看到一个关于如何查找SURF关键点和描述符并绘制它的简单演示。所有示例都显示在Python终端中，因为它只与SIFT相同

```python
>>> img = cv.imread('fly.png',0)

# Create SURF object. You can specify params here or later.
# Here I set Hessian Threshold to 400
>>> surf = cv.xfeatures2d.SURF_create(400)

# Find keypoints and descriptors directly
>>> kp, des = surf.detectAndCompute(img,None)

>>> len(kp)
 699
```

1199个关键点太多，无法在图片中显示。我们将它减少到大约50以将其绘制在图像上。在匹配时，我们可能需要所有这些功能，但现在不需要。所以我们增加了Hessian阈值。

```python
# Check present Hessian threshold
>>> print( surf.getHessianThreshold() )
400.0

# We set it to some 50000. Remember, it is just for representing in picture.
# In actual cases, it is better to have a value 300-500
>>> surf.setHessianThreshold(50000)

# Again compute keypoints and check its number.
>>> kp, des = surf.detectAndCompute(img,None)

>>> print( len(kp) )
47
```

现在小于50了。让我们在图像上绘制它。

```python
>>> img2 = cv.drawKeypoints(img,kp,None,(255,0,0),4)
>>> plt.imshow(img2),plt.show()
```

请参阅下面的结果。你可以看到SURF更像是斑点探测器。它可以探测到蝴蝶翅膀上的白色斑点。你可以使用其他图像进行测试。

![image15](https://raw.githubusercontent.com/TonyStark1997/OpenCV-Python/master/5.Feature%20Detection%20and%20Description/Image/image15.jpg)

现在我们尝试一下U-SURF，它不会检测关键点的方向。

```python
# Check upright flag, if it False, set it to True
>>> print( surf.getUpright() )
False

>>> surf.setUpright(True)

# Recompute the feature points and draw it
>>> kp = surf.detect(img,None)
>>> img2 = cv.drawKeypoints(img,kp,None,(255,0,0),4)

>>> plt.imshow(img2),plt.show()
```

结果如下。所有的关键点的朝向都是一致的。它比前面的快很多。如果你的工作对关键点的朝向没有特别的要求（如全景图拼接）等，这种方法会更快。

![image16](https://raw.githubusercontent.com/TonyStark1997/OpenCV-Python/master/5.Feature%20Detection%20and%20Description/Image/image16.jpg)

最后，我们检查描述符大小，如果它只有64-dim，则将其更改为128。

```python
# Find size of descriptor
>>> print( surf.descriptorSize() )
64

# That means flag, "extended" is False.
>>> surf.getExtended()
 False

# So we make it to True to get 128-dim descriptors.
>>> surf.setExtended(True)
>>> kp, des = surf.detectAndCompute(img,None)
>>> print( surf.descriptorSize() )
128
>>> print( des.shape )
(47, 128)
```

接下来要做的就是匹配了，我们会在后面讨论。