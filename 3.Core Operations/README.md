# 第三章：OpenCV核心操作

本章节你将学习图像的基本操作，如像素编辑、几何变换、代码优化和一些数学工具等。

更多内容请关注我的GitHub库：https://github.com/TonyStark1997，如果喜欢，star并follow我！

***

## 一、图像的基本操作

***

### 目标：

本章节您需要学习以下内容:

    * 获取并修改图像的像素值
    * 获取图像的属性
    * 设置感兴趣区域
    * 图像的拆分和合并

本节中的几乎所有操作主要与Numpy而不是OpenCV有关。使用OpenCV编写更好的优化代码需要先熟悉Numpy。

**注意：示例将在Python终端中显示，因为大多数只是单行代码**

### 1.获取并修改图像的像素值

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

### 2.获取图像的属性

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

### 3.图像ROI

有时你需要对一幅图像的特定区域进行操作。例如我们要检测一副图像中眼睛的位置，我们首先应该在图像中找到脸，再在脸的区域中找眼睛，而不是直接在一整幅图像中搜索。这样会提高程序的准确性（因为眼睛总在脸上）和性能（因为我们在很小的区域内搜索）。

ROI 也是使用 Numpy 索引来获得的。现在我们选择球的部分并把他拷贝到图像的其他区域。

```python
>>> ball = img[280:340, 330:390]
>>> img[273:333, 100:160] = ball
```

窗口将如下图所示：

![image1](https://raw.githubusercontent.com/TonyStark1997/OpenCV-Python/master/3.Core%20Operations/Image/image1.png)

### 4.图像通道的查分和合并
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

### 5.制作图像边框

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

![image2](https://raw.githubusercontent.com/TonyStark1997/OpenCV-Python/master/3.Core%20Operations/Image/image2.png)

## 二、图像的算术运算

***

### 目标：
    * 本小节你将学习对图像的几种运算，如加法、减法、按位运算等
    * 你将学习以下几个函数： cv.add(), cv.addWeighted() 

### 1.图像的加法

你可以使用OpenCV的cv.add()函数把两幅图像相加，或者可以简单地通过numpy操作添加两个图像，如res = img1 + img2。两个图像应该具有相同的大小和类型，或者第二个图像可以是标量值。

**注意：OpenCV加法和Numpy加法之间存在差异。OpenCV的加法是饱和操作，而Numpy添加是模运算。**

参考以下代码：

```python
>>> x = np.uint8([250])
>>> y = np.uint8([10])
>>> print( cv.add(x,y) ) # 250+10 = 260 => 255
[[255]]
>>> print( x+y )          # 250+10 = 260 % 256 = 4
[4]
```

这种差别在你对两幅图像进行加法时会更加明显。OpenCV 的结果会更好一点。所以我们尽量使用 OpenCV 中的函数。

### 2.图像的混合

这其实也是加法，但是不同的是两幅图像的权重不同，这就会给人一种混合或者透明的感觉。图像混合的计算公式如下：

>g(x) = (1−α)f0(x) + αf1(x)

通过修改 α 的值（0 → 1），可以实现非常炫酷的混合。

现在我们把两幅图混合在一起。第一幅图的权重是0.7，第二幅图的权重是0.3。函数cv2.addWeighted()可以按下面的公式对图片进行混合操作。

>dst = α⋅img1 + β⋅img2 + γ

这里γ取为零。

参考以下代码：

```python
img1 = cv.imread('ml.png')
img2 = cv.imread('opencv-logo.png')
dst = cv.addWeighted(img1,0.7,img2,0.3,0)
cv.imshow('dst',dst)
cv.waitKey(0)
cv.destroyAllWindows()
```

窗口将如下图显示：

![image3](https://raw.githubusercontent.com/TonyStark1997/OpenCV-Python/master/3.Core%20Operations/Image/image3.png)

### 3.图像按位操作

这里包括的按位操作有：AND，OR，NOT，XOR 等。当我们提取图像的一部分，选择非矩形ROI时这些操作会很有用（你会在后续章节中看到）。下面的例子就是教给我们如何改变一幅图的特定区域。我想把OpenCV的标志放到另一幅图像上。如果我使用图像的加法，图像的颜色会改变，如果使用图像的混合，会得到一个透明的效果，但是我不希望它透明。如果它是矩形我可以像上一章那样使用ROI。但是OpenCV标志不是矩形。所以我们可以通过下面的按位运算实现：

```python
# Load two images
img1 = cv.imread('messi5.jpg')
img2 = cv.imread('opencv-logo-white.png')
# I want to put logo on top-left corner, So I create a ROI
rows,cols,channels = img2.shape
roi = img1[0:rows, 0:cols ]
# Now create a mask of logo and create its inverse mask also
img2gray = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
ret, mask = cv.threshold(img2gray, 10, 255, cv.THRESH_BINARY)
mask_inv = cv.bitwise_not(mask)
# Now black-out the area of logo in ROI
img1_bg = cv.bitwise_and(roi,roi,mask = mask_inv)
# Take only region of logo from logo image.
img2_fg = cv.bitwise_and(img2,img2,mask = mask)
# Put logo in ROI and modify the main image
dst = cv.add(img1_bg,img2_fg)
img1[0:rows, 0:cols ] = dst
cv.imshow('res',img1)
cv.waitKey(0)
cv.destroyAllWindows()
```

窗口将如下图显示。左面的图像是我们创建的模板，右边的是最终结果。为了帮助大家理解，我把上面程序的中间结果也显示了出来，特别是img1_bg和img2_fg。

![image4](https://raw.githubusercontent.com/TonyStark1997/OpenCV-Python/master/3.Core%20Operations/Image/image4.png)

## 三、程序性能评估及优化
***

### 目标：

在图像处理中，由于每秒需要处理大量操作，因此处理图像的代码必须不仅要能给出正确的结果，同时还必须要快。所以在本小节中，你将学习：

    * 衡量代码的性能。
    * 一些优化代码性能的技巧。
    * 你将学习以下几个函数：cv.getTickCount, cv.getTickFrequency

除了OpenCV库之外，Python还提供了一个time模块，有助于测量执行时间。另一个profile模块可以获得有关代码的详细报告，例如代码中每个函数所花费的时间，调用函数的次数等。如果你使用的是IPython，所有这些功能都以一个有好的方式整合到一起。我们将看到一些重要的内容，有关更多详细信息，请查看“其他资源”部分中的链接。

### 1.使用 OpenCV 衡量程序效率

cv.getTickCount函数返回参考事件（如机器开启时刻）到调用此函数的时钟周期数。因此，如果在函数执行之前和之后都调用它，则会获得用于执行函数的时钟周期数。

cv.getTickFrequency函数返回时钟周期的频率，或每秒钟的时钟周期数。所以，要想获得函数的执行时间，您可以执行以下操作：

```python
e1 = cv.getTickCount()
# your code execution
e2 = cv.getTickCount()
time = (e2 - e1)/ cv.getTickFrequency()
```

我们将展示以下示例示例，下面的例子使用从5到49几个不同大小的核进行中值滤波。（不要考虑结果会是什么样的，这不是我们的目的）：

```python
img1 = cv.imread('messi5.jpg')
e1 = cv.getTickCount()
for i in xrange(5,49,2):
    img1 = cv.medianBlur(img1,i)
e2 = cv.getTickCount()
t = (e2 - e1)/cv.getTickFrequency()
print( t )
# Result I got is 0.521107655 seconds
```

**注意：您可以使用time模块的函数执行相同操作来替代cv.getTickCount，使用time.time()函数,然后取两次结果的时间差。**

### 2.OpenCV中的默认优化

许多OpenCV的功能都使用SSE2，AVX等进行了优化。它还包含了一些未经优化的代码。因此，如果我们的系统支持这些功能，我们应该利用它们（几乎所有现代处理器都支持它们），编译时是默认启用优化。因此，OpenCV运行的代码就是已优化代码（如果已启用），否则运行未优化代码。 您可以使用cv.useOptimized()来检查它是否已启用/禁用优化，并使用cv.setUseOptimized())来启用/禁用它。让我们看一个简单的例子。

```python
# check if optimization is enabled
In [5]: cv.useOptimized()
Out[5]: True
In [6]: %timeit res = cv.medianBlur(img,49)
10 loops, best of 3: 34.9 ms per loop
# Disable it
In [7]: cv.setUseOptimized(False)
In [8]: cv.useOptimized()
Out[8]: False
In [9]: %timeit res = cv.medianBlur(img,49)
10 loops, best of 3: 64.1 ms per loop
```

优化的中值滤波比未优化的版本快2倍。如果检查其来源，您可以看到中值滤波是经过SIMD优化的。因此，你可以使用它来在代码顶部启用优化（请记住它默认启用）。

### 3.检测IPython中的性能

有时您可能需要比较两个类似操作的性能。IPython提供了一个魔术命令timeit来执行此操作。 它可以让代码运行几次以获得更准确的结果。它们也适用于测量单行代码。

例如，你想知道以下哪个加法操作更快吗？
>x = 5; y = x ** 2
>x = 5; y = x * x
>x = np.uint8（[5]）; y = x * x
>y = np.square（x）
我们将在IPython shell中使用timeit得到答案。

```python
In [10]: x = 5
In [11]: %timeit y=x**2
10000000 loops, best of 3: 73 ns per loop
In [12]: %timeit y=x*x
10000000 loops, best of 3: 58.3 ns per loop
In [15]: z = np.uint8([5])
In [17]: %timeit y=z*z
1000000 loops, best of 3: 1.25 us per loop
In [19]: %timeit y=np.square(z)
1000000 loops, best of 3: 1.16 us per loop
```

你可以看到，x = 5; y = x * x是最快的，与Numpy相比快了约20倍。如果您也考虑创建阵列，它可能会快达100倍。很酷对不对 （Numpy开发者正在研究这个问题）

**注意：Python标量操作比Numpy标量操作更快。因此对于包含一个或两个元素的操作，Python标量优于Numpy数组。当阵列的大小稍大时，Numpy会占据优势。**

我们将再尝试一个例子。 这次，我们将比较同一图像的cv.countNonZero()和np.count_nonzero()的性能。
```python
In [35]: %timeit z = cv.countNonZero(img)
100000 loops, best of 3: 15.8 us per loop
In [36]: %timeit z = np.count_nonzero(img)
1000 loops, best of 3: 370 us per loop
```

你可以看到，OpenCV的执行性能比Numpy快将近25倍。

**注意：通常，OpenCV函数比Numpy函数更快。因此对于相同的操作，OpenCV功能是首选。但是可能也有例外，尤其是当使用Numpy对视图而不是复制数组时。**

### 4.更多的IPython命令

还有几个魔法命令可以用来检测程序的执行效率，profiling，line profiling，memory measurement等。他们都有完善的文档。所以这里只提供了超链接。感兴趣的读者可以自己学习一下。

### 5.性能优化技术

有几种技术和编码方法可以利用Python和Numpy的最大性能。此处仅注明相关的内容，并提供重要来源的链接。这里要注意的主要是，首先尝试以简单的方式实现算法。一旦工作，对其进行分析，找到瓶颈并进行优化。

1. 尽量避免在Python中使用循环，尤其是双层/三层嵌套循环等。它们本身就很慢。
2. 将算法/代码尽量使用向量化操作，因为Numpy和OpenCV针对向量运算进行了优化。
3. 利用高速缓存一致性。
4. 除非需要，否则不要复制数组。尝试使用视图去替代复制数组。数组复制是一项非常浪费资源的操作。

即使在完成所有这些操作之后，如果您的代码仍然很慢，或者使用大型循环是不可避免的，请使用其他库（如Cython）来加快速度。
