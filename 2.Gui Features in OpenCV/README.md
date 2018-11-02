# 第二章：OpenCV中的Gui相关功能

本章节你将学习如何显示和保存图像和视频、控制鼠标事件和创建轨迹栏。

更多内容请关注我的GitHub库：https://github.com/TonyStark1997，如果喜欢，star并follow我！

***

## 一、图像处理入门

***

### 目标：

* 在本小节你将学习如何读取图像、如何显示图像，还有如何保存图像
* 你将学习以下几个函数：cv.imread(), cv.imshow() , cv.imwrite()
* 之后，你将学习如何试用Matplotlib显示图像

### 1. 读取图像
使用cv.imread()函数用于读取图像，其中函数的第一个参数为要读取的图像名称，此图像应该处在Python代码文件的工作目录中，或者应给出完整的文件路径。第二个参数是一个标志，以下几种参数分别指定了应该读取图像的方式。
* cv.IMREAD_COLOR：以彩色模式加载图像，任何图像的透明度都将被忽略。这是默认参数。
* cv.IMREAD_GRAYSCALE：以灰度模式加载图像。
* cv.IMREAD_UNCHANGED：包括alpha通道的加载图像模式。

**注意：或者你可以简单的传递1、0或者-1来替代上面三个标志。**

参考以下代码：

```python
import numpy as np
import cv2 as cv
# Load an color image in grayscale
img = cv.imread('messi5.jpg',0)
```

**注意：如果加载图像的路径有错误，它并不会报错，而是返回给你一个None值。**

### 2. 显示图像
使用cv.imshow()函数用于在窗口中显示图像，窗口会自动适合图像大小。其中函数的第一个参数是窗口的名称，以字符串类型表示。第二个参数是要加载的图像。你可以显示多个图像窗口，只要它们的窗口名称不同就可以。

参考以下代码：

```python
cv.imshow('image',img)
cv.waitKey(0)
cv.destroyAllWindows()
```

窗口将如下图所示：

![image1](https://raw.githubusercontent.com/TonyStark1997/OpenCV-Python/master/2.Gui%20Features%20in%20OpenCV/image/image1.png)

cv.waitKey()是一个键盘事件函数，它的参数是以毫秒为单位的时间。该函数等待参数时间，如果时间之内有键盘事件触发则程序继续，如果函数参数设置为0，则无限时间的等待键盘事件触发。它也可以设置为检测指定按键的触发，比如等待按键a的触发，我们将在下面讨论。

**注意：这个函数除了可以等待键盘事件的触发之外还可以处理很多其他的GUI事件，所以你必须把它放在显示图像函数之后。**

cv.destroyAllWindow()函数用于关闭我们所创建的所有显示图像的窗口，如果想要关闭任何特定的窗口，请使用cv.destroyWindow()函数，其中把要关闭的窗口名称作为参数传递。

**注意：一种特殊的情况是，你也可以先创建一个窗口，之后再加载图像。这种情况下你可以自行决定窗口的大小，你可以使用cv.nameWindow()函数进行窗口大小的调整。默认函数参数是cv.WINDOW_AUTOSIZE，你可以将其改成cv.WINDOW_NORMAL，这样你就可以自行调整窗口大小了。当图像尺寸太大或者需要添加轨迹条时，调整窗口大小将会非常有用。**

参考以下代码：

```python
cv.namedWindow('image', cv.WINDOW_NORMAL)
cv.imshow('image',img)
cv.waitKey(0)
cv.destroyAllWindows()
```

### 3. 保存图像
使用cv.imwrite()函数用于保存图像。其中第一个函数是文件名，第二个函数是你要保存的图像。

参考一下代码：

```python
cv.imwrite('messigray.png',img)
```

这行代码将Python代码文件的工作目录中保存PNG格式的图像。

### 4. 总结以上三条
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

**注意：如果你使用的是64位计算机，则必须将k = cv.waitKey(0)修改为：k = cv.waitKey(0) ＆ 0xFF**

### 5.使用Matplotlib

Matplotlib是Python的绘图库，为你提供各种绘图方法。 你将在即将发表的文章中看到它们。 在这里，你将学习如何使用Matplotlib显示图像。 你可以使用Matplotlib缩放图像，保存等。

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

![image2](https://raw.githubusercontent.com/TonyStark1997/OpenCV-Python/master/2.Gui%20Features%20in%20OpenCV/image/image2.png)

Matplotlib提供了大量的绘图选项。有关更多详细信息，请参阅Matplotlib文档。我们将在学习过程中看到一些。

**注意：OpenCV加载的彩色图像处于BGR模式。但Matplotlib以RGB模式显示。因此，如果使用OpenCV读取图像，则Matplotlib中的彩色图像将无法正确显示。请参阅练习了解更多详情。**

## 二、视频处理入门

***

### 目标：

* 在本小节你将学习读取视频、显示视频和保存视频
* 你将学习用摄像头捕获视频并显示
* 你将学习以下几个函数：cv.VideoCapture(), cv.VideoWriter()

### 1.用摄像头捕获视频

通常，我们需要用摄像头来捕获直播画面。OpenCV为此提供了一些非常简单的函数接口。下面我们来尝试用摄像头来捕获视频画面（此刻我使用的是笔记本电脑的内置网络摄像头）并将画面转化成灰度图像显示出来。这只是一项非常简单的任务。

如果要捕获视频，首先要做的是创建一个VideoCapture对象，它的参数可以是设备索引或者是视频文件的名称。设备索引就是指设备所对应的设备号，通常你只连接一个摄像头，所以参数只传递0（或-1）就可以。你可以传递参数1来选择你连接的第二个摄像头，以此类推。之后，你需要逐帧捕获并显示。最后，不要忘记关闭捕获。

参考一下代码：

```python
import numpy as np
import cv2 as cv
cap = cv.VideoCapture(0)
while(True):
# Capture frame-by-frame
ret, frame = cap.read()
# Our operations on the frame come here
gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
# Display the resulting frame
cv.imshow('frame',gray)
if cv.waitKey(1) & 0xFF == ord('q'):
break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
```

cap.read()返回一个bool值（True / False）。如果读取帧正确，则它将为True。因此，你可以通过检查此返回值来检查视频的结尾。

有时可能不能成功的初始化摄像头设备。这种情况下上面的代码会报错。你可以使用cap.isOpened()，来检查是否成功初始化了。如果返回值是True，那就没有问题。否则就要使用函数 cap.open()。

你还可以使用cap.get(propld)方法访问此视频的某些功能，其中propId是0到18之间的数字。每个数字表示视频的属性（如果它适用于该视频），完整详细的信息你可以在这里看到：[cv::VideoCapture::get()](https://docs.opencv.org/3.4.1/d8/dfe/classcv_1_1VideoCapture.html#aa6480e6972ef4c00d74814ec841a2939).其中一些值可以使用cap.set(propId，value)进行修改。其中参数value是你想要的新值。

例如，我可以通过cap.get(cv.CAP_PROP_FRAME_WIDTH)和cap.get(cv.CAP_PROP_FRAME_HEIGHT)来分别检查帧宽和高度。它返回给我默认值640x480。但如果我想将其修改为320x240，只需使用ret = cap.set(cv.CAP_PROP_FRAME_WIDTH，32)）和ret = cap.set(cv.CAP_PROP_FRAME_HEIGHT，240)后即可改变。

**注意：如果收到报错信息，请确保其他使用摄像头的程序在正常工作（如Linux中的Cheese）。**

### 2.播放视频文件

它与从相机捕获视频图像原理相同，只需将设备索引更改为视频文件的名字。同时在显示帧时，请给cv.waitKey()函数传递适当的时间参数。如果它太小，视频将非常快，如果它太高，视频将会很慢（这就是你可以用慢动作显示视频）。在正常情况下，25毫秒就可以了。

参考以下代码：

```python
import numpy as np
import cv2 as cv
cap = cv.VideoCapture('vtest.avi')
while(cap.isOpened()):
ret, frame = cap.read()
gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
cv.imshow('frame',gray)
if cv.waitKey(1) & 0xFF == ord('q'):
break
cap.release()
cv.destroyAllWindows()
```

**注意：确保安装了正确版本的ffmpeg或gstreamer。 有时候，使用Video Capture是一个令人头疼的问题，主要原因是错误安装了ffmpeg/gstreamer。**

### 3.保存视频

目前为止我们可以捕获视频，并且逐帧显示并进行处理，现在我们希望保存该视频。对于图片的话是很简单进行保存的，但是对于视频，我们还需要做更多的工作。

这次，我们创建一个VideoWriter对象，我们应该指定输出文件名（例如：output.avi）。然后我们应该指定FourCC代码（下一段中的详细信息）。然后应该传递每秒帧数（fps）和帧大小。最后一个是isColor标志。如果是True，则每一帧是彩色图像，否则每一帧是灰度图像。

[FourCC](https://en.wikipedia.org/wiki/FourCC)是用于指定视频编解码器的4字节代码。可以在[fourcc.org](http://www.fourcc.org/codecs.php)中找到可用代码列表，它取决于平台。以下编解码器对我来说是有用的：
* 在Fedora中：DIVX，XVID，MJPG，X264，WMV1，WMV2。（XVID更为可取.MJPG会产生高大小的视频.X264提供非常小的视频）
* 在Windows中：DIVX（更多要测试和添加）
* 在OSX中：MJPG（.mp4），DIVX（.avi），X264（.mkv）。

在从相机捕获图像之后，在垂直方向上翻转每一帧之后逐帧保存。

参考以下代码：

```python
import numpy as np
import cv2 as cv
cap = cv.VideoCapture(0)
# Define the codec and create VideoWriter object
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('output.avi',fourcc, 20.0, (640,480))
while(cap.isOpened()):
ret, frame = cap.read()
if ret==True:
frame = cv.flip(frame,0)
# write the flipped frame
out.write(frame)
cv.imshow('frame',frame)
if cv.waitKey(1) & 0xFF == ord('q'):
break
else:
break
# Release everything if job is finished
cap.release()
out.release()
cv.destroyAllWindows()
```

## 三、在OpenCV中的绘制函数

***

### 目标：

* 在本小节你将学习用OpenCV绘制不同的几何图形
* 你将学习以下几个函数：cv.line(), cv.circle() , cv.rectangle(), cv.ellipse(), cv.putText() 

在上述所有函数中你将看到以下几个常见的参数：

* img：用于设置要绘制形状的图像
* color：用于设置绘制图案的颜色。对于BGR图像，将其用元组传递，例如：（255,0,0）为蓝色。对于灰度图像，只需传递标量值。
* thickness：用于设置线条或圆形等的厚度。如果是- 1则在图案内生成闭合图案并填充颜色。这个参数的默认厚度为1。
* lineType：用于设置线条的类型，有8型连接，抗锯齿等。默认情况是8型连接。cv2.LINE_AA为抗锯齿，这样看起来会非常平滑。

### 1. 绘制直线

要绘制线条，你需要传递线条的起点和终点坐标。我们将创建一个黑色图像，并在其上从左上角到右下角绘制一条蓝线。
参考以下代码：

```python
import numpy as np
import cv2 as cv
# Create a black image
img = np.zeros((512,512,3), np.uint8)
# Draw a diagonal blue line with thickness of 5 px
cv.line(img,(0,0),(511,511),(255,0,0),5)
```

### 2. 绘制矩形

要绘制矩形，你需要传递矩形的左上角和右下角的坐标。这次我们将在图像的右上角绘制一个绿色矩形。
参考一下代码：

```python
cv.rectangle(img,(384,0),(510,128),(0,255,0),3)
```

### 3. 绘制圆形

要绘制圆形，你需要传递其圆点坐标和半径，我们将在上面绘制的矩形内绘制一个圆。
参考一下代码：

```python
cv.circle(img,(447,63), 63, (0,0,255), -1)
```

### 4. 绘制椭圆

要绘制椭圆，我们需要传递四个参数，第一个是椭圆中心位置（x，y），第二个是长轴长度和断轴长度（a，b），第三个是椭圆在逆时针方向上的旋转角度，第四个是startAngle和endAngle表示从主轴顺时针方向测量的椭圆弧的起点和终点，即给出值0和360给出完整的椭圆，给出值180则画出半个椭圆。
参考一下代码：

```python
cv.ellipse(img,(256,256),(100,50),0,0,180,255,-1)

```

### 5. 绘制多边形

要绘制多边形，首先需要顶点坐标。 将这些点转换为ROWSx1x2的数组，其中ROWS是顶点数，它应该是int32类型。在这里，我们绘制一个带有四个黄色顶点的小多边形。
参考一下代码：

```python
pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
pts = pts.reshape((-1,1,2))
cv.polylines(img,[pts],True,(0,255,255))
```

**注意：如果第三个参数为False，则绘制所有点的相连图形而不是闭合图形。cv.polylines（）可用于绘制多条线。 只需创建要绘制的所有行的列表并将其传递给函数。 所有线条都将单独绘制。 绘制一组行比为每行调用cv.line（）要好得多，速度更快。**

### 6. 向图像中添加文字

要将文本放入图像中，你需要传递以下几个参数：第一个是你要写入的文本数据，第二个是你要放置的位置（即文本数据的左下角），第三个是字体类型（检查cv.putText（）文档以获取支持的字体），的四个是字体大小，之后还有一些常规的参数，比如颜色、粗细、线型等，为了更好看，建议使用lineType = cv.LINE_AA作为线型参数的值。

我们将在图像上编写白色的OpenCV字样。
参考以下代码：

```python
font = cv.FONT_HERSHEY_SIMPLEX
cv.putText(img,'OpenCV',(10,500), font, 4,(255,255,255),2,cv.LINE_AA)
```

### 7. 结果

是时候看看我们所绘制图案的结果了，正如文章之间所讲述的那样，通过显示图像将上面六个绘制结果显示出来。
窗口将如下图所示：

![image3](https://raw.githubusercontent.com/TonyStark1997/OpenCV-Python/master/2.Gui%20Features%20in%20OpenCV/image/image3.png)

## 四、鼠标作为画笔

***

### 目标：

* 在本小节你将学习用OpenCV控制鼠标事件
* 你将学习以下几个函数：cv.setMouseCallback()

### 一个简单的示例

这里我们来创建一个简单的程序，他会在图片上你双击过的位置绘制一个圆圈。首先我们来创建一个鼠标事件回调函数，但鼠标事件发生是他就会被执行。鼠标事件可以是鼠标上的任何动作，比如左键按下，左键松开，左键双击等。我们可以通过鼠标事件获得与鼠标对应的图片上的坐标。根据这些信息我们可以做任何我们想做的事。你可以通过执行下列代码查看所有被支持的鼠标事件：

```python
import cv2 as cv
events = [i for i in dir(cv) if 'EVENT' in i]
print( events )
```

所有的鼠标事件回调函数都有一个统一的格式，他们所不同的地方仅仅是被调用后的功能。我们只需要鼠标事件回调函数做一件事：在双击过的地方绘制一个圆形。下面是代码，可以通过注释理解代码:

```python
import numpy as np
import cv2 as cv
# mouse callback function
def draw_circle(event,x,y,flags,param):
if event == cv.EVENT_LBUTTONDBLCLK:
cv.circle(img,(x,y),100,(255,0,0),-1)
# Create a black image, a window and bind the function to window
img = np.zeros((512,512,3), np.uint8)
cv.namedWindow('image')
cv.setMouseCallback('image',draw_circle)
while(1):
cv.imshow('image',img)
if cv.waitKey(20) & 0xFF == 27:
break
cv.destroyAllWindows()
```

### 一个更高级的示例
现在我们来创建一个更好的程序。这次我们的程序要完成的任务是根据我们选择的模式在拖动鼠标时绘制矩形或者是圆圈（就像画图程序中一样）。所以我们的回调函数包含两部分，一部分画矩形，一部分画圆圈。这是一个典型的例子他可以帮助我们更好理解与构建人机交互式程序，比如物体跟踪，图像分割等。
参考以下代码：

```python
import cv2 as cv
drawing = False # true if mouse is pressed
mode = True # if True, draw rectangle. Press 'm' to toggle to curve
ix,iy = -1,-1
# mouse callback function
def draw_circle(event,x,y,flags,param):
global ix,iy,drawing,mode
if event == cv.EVENT_LBUTTONDOWN:
drawing = True
ix,iy = x,y
elif event == cv.EVENT_MOUSEMOVE:
if drawing == True:
if mode == True:
cv.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
else:
cv.circle(img,(x,y),5,(0,0,255),-1)
elif event == cv.EVENT_LBUTTONUP:
drawing = False
if mode == True:
cv.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
else:
cv.circle(img,(x,y),5,(0,0,255),-1)
```

接下来，我们必须将此鼠标回调函数绑定到OpenCV窗口。在主循环中，我们应该把按键'm'设置为切换绘制矩形还是圆形。
参考以下代码：

```python
img = np.zeros((512,512,3), np.uint8)
cv.namedWindow('image')
cv.setMouseCallback('image',draw_circle)
while(1):
cv.imshow('image',img)
k = cv.waitKey(1) & 0xFF
if k == ord('m'):
mode = not mode
elif k == 27:
break
cv.destroyAllWindows()
```

## 五、轨迹栏作为调色板

***

### 目标：

* 在本小节你将学习把轨道栏绑定到OpenCV窗口中
* 你将学习以下几个函数：cv.getTrackbarPos(), cv.createTrackbar() 

### 代码示例

在这里，我们将创建一个简单的应用程序，完成显示指定的颜色。你有一个显示颜色的窗口和三个轨道栏，分别用于指定B，G，R各颜色。你可以滑动轨迹栏并相应地更改窗口所显示的颜色。默认情况下，初始颜色将设置为黑色。

对于cv.getTrackbarPos()函数，第一个参数是轨道栏名称，第二个参数是它所附加的窗口名称，第三个参数是默认值，第四个参数是最大值，第五个参数是执行的回调函数每次轨迹栏值都会发生变化。回调函数始终具有默认参数，即轨迹栏位置。在我们的例子中，函数什么都不做，所以我们简单地跳过。

轨迹栏的另一个重要应用是将其用作按钮或开关。默认情况下，OpenCV没有按钮功能。因此，你可以使用跟踪栏来获得此类功能。在我们的应用程序中，我们创建了一个开关，其中应用程序仅在开关打开时有效，否则屏幕始终为黑色。
参考一下代码：

```python
import numpy as np
import cv2 as cv
def nothing(x):
pass
# Create a black image, a window
img = np.zeros((300,512,3), np.uint8)
cv.namedWindow('image')
# create trackbars for color change
cv.createTrackbar('R','image',0,255,nothing)
cv.createTrackbar('G','image',0,255,nothing)
cv.createTrackbar('B','image',0,255,nothing)
# create switch for ON/OFF functionality
switch = '0 : OFF \n1 : ON'
cv.createTrackbar(switch, 'image',0,1,nothing)
while(1):
cv.imshow('image',img)
k = cv.waitKey(1) & 0xFF
if k == 27:
break
# get current positions of four trackbars
r = cv.getTrackbarPos('R','image')
g = cv.getTrackbarPos('G','image')
b = cv.getTrackbarPos('B','image')
s = cv.getTrackbarPos(switch,'image')
if s == 0:
img[:] = 0
else:
img[:] = [b,g,r]
if cv.waitKey(1) & 0xFF == ord('q'):
break

cv.destroyAllWindows()
```

窗口将如下图所示：

![image4](https://raw.githubusercontent.com/TonyStark1997/OpenCV-Python/master/2.Gui%20Features%20in%20OpenCV/image/image4.png)

