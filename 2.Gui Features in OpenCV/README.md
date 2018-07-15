# 第二章：OpenCV核心操作
本章节你将学习图像的基本操作，如像素编辑、几何变换、代码优化和一些数学工具等。
***
## 一、图像处理入门
***
### 目标：
* 在本小节你将学习如何读取图像、如何显示图像，还有如何保存图像
* 你将学习以下几个函数：cv.imread(), cv.imshow() , cv.imwrite()
* 之后，您将学习如何试用Matplotlib显示图像

### 使用OpenCV
1. 读取图像
使用cv.imread()函数用于读取图像，其中函数的第一个参数为要读取的图像名称，此图像应该处在Python代码文件的工作目录中，或者应给出完整的文件路径。第二个参数是一个标志，以下几种参数分别指定了应该读取图像的方式。
* cv.IMREAD_COLOR：以彩色模式加载图像，任何图像的透明度都将被忽略。这是默认参数。
* cv.IMREAD_GRAYSCALE：以灰度模式加载图像。
* cv.IMREAD_UNCHANGED：包括alpha通道的加载图像模式。
**注意：或者您可以简单的传递1、0或者-1来替代上面三个标志。**

参考以下代码：
```python
import numpy as np
import cv2 as cv
# Load an color image in grayscale
img = cv.imread('messi5.jpg',0)
```
**注意：如果加载图像的路径有错误，它并不会报错，而是返回给你一个None值。**

2. 显示图像
使用cv.imshow()函数用于在窗口中显示图像，窗口会自动适合图像大小。其中函数的第一个参数是窗口的名称，以字符串类型表示。第二个参数是要加载的图像。你可以显示多个图像窗口，只要它们的窗口名称不同就可以。

参考以下代码：
```python
cv.imshow('image',img)
cv.waitKey(0)
cv.destroyAllWindows()
```
窗口将如下图所示：

cv.waitKey()是一个键盘事件函数，它的参数是以毫秒为单位的时间。该函数等待参数时间，如果时间之内有键盘事件触发则程序继续，如果函数参数设置为0，则无限时间的等待键盘事件触发。它也可以设置为检测指定按键的触发，比如等待按键a的触发，我们将在下面讨论。
**注意：这个函数除了可以等待键盘事件的触发之外还可以处理很多其他的GUI事件，所以你必须把它放在显示图像函数之后。**

cv.destroyWindow()函数用于关闭我们所创建的所有显示图像的窗口，如果想要关闭任何特定的窗口，请使用cv.destroyWindow()函数，其中把要关闭的窗口名称作为参数传递。
**注意：一种特殊的情况是，你也可以先创建一个窗口，之后再加载图像。种种情况下你可以自行决定窗口的大小，你可以使用cv.nameWindow()函数进行窗口大小的调整。默认函数参数是cv.WINDOW_AUTOSIZE，你可以将其改成cv.WINDOW_NORMAL，这样你就可以自行调整窗口大小了。当图像尺寸太大或者需要添加轨迹条时，调整窗口大小将会非常有用。**

参考以下代码：
```python
cv.namedWindow('image', cv.WINDOW_NORMAL)
cv.imshow('image',img)
cv.waitKey(0)
cv.destroyAllWindows()
```
3. 保存图像
使用cv.imwrite()函数用于保存图像。其中第一个函数是文件名，第二个函数是你要保存的图像。

参考一下代码：
```python
cv.imwrite('messigray.png',img)
```
这行代码将Python代码文件的工作目录中保存PNG格式的图像。
4. 总结以上三条
下面的代码程序将加载灰度图像，显示图像，如果按's'并退出则保存图像，或者按ESC键直接退出而不保存。

参考一下代码：
```python
import numpy as np
import cv2 as cv
img = cv.imread('messi5.jpg',0)
cv.imshow('image',img)
k = cv.waitKey(0)
if k == 27:         # wait for ESC key to exit
cv.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
cv.imwrite('messigray.png',img)
cv.destroyAllWindows()
```
**注意：如果您使用的是64位计算机，则必须将k = cv.waitKey(0)修改为：k = cv.waitKey(0) ＆ 0xFF**

### 使用Matplotlib
Matplotlib是Python的绘图库，为您提供各种绘图方法。 您将在即将发表的文章中看到它们。 在这里，您将学习如何使用Matplotlib显示图像。 您可以使用Matplotlib缩放图像，保存等。

参考以下代码：
```python
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread('messi5.jpg',0)
plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()
```
窗口将如下图所示：

Matplotlib提供了大量的绘图选项。有关更多详细信息，请参阅Matplotlib文档。我们将在学习过程中看到一些。

**注意：OpenCV加载的彩色图像处于BGR模式。但Matplotlib以RGB模式显示。因此，如果使用OpenCV读取图像，则Matplotlib中的彩色图像将无法正确显示。请参阅练习了解更多详情。**

## 二、视频处理入门
***

## 三、在OpenCV中的绘制函数
***

## 四、鼠标作为画笔
***

## 五、轨迹栏作为调色板
***
