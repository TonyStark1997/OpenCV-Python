# 第三章：OpenCV核心操作

本章节你将学习图像的基本操作，如像素编辑、几何变换、代码优化和一些数学工具等。
***

## 图像的基本操作
***

### 目标：

本章节您需要学习以下内容:

* 获取并修改图像的像素值
* 获取图像的属性
* 设置感兴趣区域
* 图像的拆分和合并

本节中的几乎所有操作主要与Numpy而不是OpenCV有关。使用OpenCV编写更好的优化代码需要先熟悉Numpy。

**注意：示例将在Python终端中显示，因为大多数只是单行代码**

### 获取并修改图像的像素值

首先我们先加载一章彩色图像：

```python
>>> import numpy as np
>>> import cv2 as cv
>>> img = cv.imread('messi5.jpg')
```

之后你可以通过行和列的坐标值获取该像素点的像素值。对于BGR图像，它返回一个蓝色，绿色，红色值的数组。对于灰度图像，仅返回相应的强度值。

```python
>>> px = img[100,100]
>>> print( px )
[157 166 200]
# accessing only blue pixel
>>> blue = img[100,100,0]
>>> print( blue )
157
```

你可以用同样的方法修改像素点的像素值：

```python
>>> img[100,100] = [255,255,255]
>>> print( img[100,100] )
[255 255 255]
```

**注意：Numpy是一个用于快速阵列计算的优化库。因此，简单地访问每个像素值并对其进行修改将非常缓慢，并且不推荐这样做。上述方法通常用于选择数组的区域，比如前5行和后3列。对于单个像素访问，Numpy数组方法array.item()和array.itemset()被认为是更好的选择，但它们总是返回标量。如果要获取所有B，G，R值，则需要单独调用array.item()。**

更好的像素获取和编辑方法：

```python
# accessing RED value
>>> img.item(10,10,2)
59
# modifying RED value
>>> img.itemset((10,10,2),100)
>>> img.item(10,10,2)
100
```

### 获取图像的属性

图像属性包括行数，列数和通道数，图像数据类型，像素数等。

使用img.shape可以获取图像的形状。它返回一组行，列和通道的元组（如果图像是彩色的）：

```python
>>> print( img.shape )
(342, 548, 3)
```

**注意：如果图像是灰度图像，则返回的元组仅包含行数和列数，因此检查加载的图像是灰度还是颜色是一种很好的方法。**

使用img.size获取的像素总数：

```python
>>> print( img.size )
562248
```

使用img.dtype获取图像数据类型：

```python
>>> print( img.dtype )
uint8
```

**注意：img.dtype在调试时非常重要，因为OpenCV-Python代码中的大量错误是由无效的数据类型引起的。**

### 图像ROI

有时你需要对一幅图像的特定区域进行操作。例如我们要检测一副图像中眼睛的位置，我们首先应该在图像中找到脸，再在脸的区域中找眼睛，而不是直接在一整幅图像中搜索。这样会提高程序的准确性（因为眼睛总在脸上）和性能（因为我们在很小的区域内搜索）。

ROI 也是使用 Numpy 索引来获得的。现在我们选择球的部分并把他拷贝到图像的其他区域。

```python
>>> ball = img[280:340, 330:390]
>>> img[273:333, 100:160] = ball
```

窗口将如下图所示：


### 图像通道的查分和合并
有时需要在B，G，R通道图像上单独工作。在这种情况下，需要将BGR图像分割为单个通道。或者在其他情况下，可能需要将这些单独的通道合并到BGR图像。您可以通过以下方式完成：

```python
>>> b,g,r = cv.split(img)
>>> img = cv.merge((b,g,r))
```

或者：

```python
>>> b = img[:,:,0]
```

假设您要将所有红色像素设置为零，则无需先拆分通道。使用Numpy索引更快：

```python
>>> img[:,:,2] = 0
```

**注意：cv.split()是一项代价高昂的操作（就消耗时间而言）。所以只有在你需要时才这样做。否则使用Numpy索引。**

### 制作图像边框

如果要在图像周围创建边框，比如相框，可以使用cv.copyMakeBorder()。但它有更多卷积运算，零填充等应用。该函数需要以下参数：

* src 输入图像
* top, bottom, left, right 对应边界的像素数目。
* borderType 要添加那种类型的边界，类型如下：
– cv2.BORDER_CONSTANT 添加有颜色的常数值边界，还需要下一个参数（value）。
– cv2.BORDER_REFLECT 边界元素的镜像。比如: fedcba|abcde-fgh|hgfedcb
– cv2.BORDER_REFLECT_101 or cv2.BORDER_DEFAULT跟上面一样，但稍作改动。例如: gfedcb|abcdefgh|gfedcba
– cv2.BORDER_REPLICATE 重复最后一个元素。例如: aaaaaa|abcdefgh|hhhhhhh
– cv2.BORDER_WRAP 不知道怎么说了, 就像这样: cdefgh|abcdefgh|abcdefg
* value 边界颜色，如果边界的类型是 cv2.BORDER_CONSTANT

下面是一个示例代码，演示了所有这些边框类型，以便更好地理解：

```python
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
BLUE = [255,0,0]
img1 = cv.imread('opencv-logo.png')
replicate = cv.copyMakeBorder(img1,10,10,10,10,cv.BORDER_REPLICATE)
reflect = cv.copyMakeBorder(img1,10,10,10,10,cv.BORDER_REFLECT)
reflect101 = cv.copyMakeBorder(img1,10,10,10,10,cv.BORDER_REFLECT_101)
wrap = cv.copyMakeBorder(img1,10,10,10,10,cv.BORDER_WRAP)
constant= cv.copyMakeBorder(img1,10,10,10,10,cv.BORDER_CONSTANT,value=BLUE)
plt.subplot(231),plt.imshow(img1,'gray'),plt.title('ORIGINAL')
plt.subplot(232),plt.imshow(replicate,'gray'),plt.title('REPLICATE')
plt.subplot(233),plt.imshow(reflect,'gray'),plt.title('REFLECT')
plt.subplot(234),plt.imshow(reflect101,'gray'),plt.title('REFLECT_101')
plt.subplot(235),plt.imshow(wrap,'gray'),plt.title('WRAP')
plt.subplot(236),plt.imshow(constant,'gray'),plt.title('CONSTANT')
plt.show()
```

窗口将如下图所示（图像与matplotlib一起显示。因此RED和BLUE通道将互换）：


## 图像的算术运算
***

### 目标：
* 本小节你将学习对图像的几种运算，如加法、减法、按位运算等
* 你将学习以下几个函数： cv.add(), cv.addWeighted() 



## 程序性能评估及优化
***

### 目标：

