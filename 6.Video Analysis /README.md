# 第六章：视频分析

本章节你将学习Meanshift和Camshift、光流和背景消除等OpenCV视频分析的相关内容。

更多内容请关注我的[GitHub库:TonyStark1997](https://github.com/TonyStark1997)，如果喜欢，star并follow我！

***

## 一、Meanshift和Camshift

***

### 目标：

本章节你需要学习以下内容:

    *我们将学习Meanshift和Camshift算法来查找和跟踪视频中的对象。

### 1、Meanshift

Meanshift 算法的基本原理是和很简单的。假设我们有一堆点（比如直方图反向投影得到的点），和一个小的圆形窗口，我们要完成的任务就是将这个窗口移动到最大灰度密度处（或者是点最多的地方）。就像下面给出的图像所示：

![image1](https://raw.githubusercontent.com/TonyStark1997/OpenCV-Python/master/6.Video%20Analysis%20/Image/image1.jpg)

初始窗口是蓝色的“C1”，它的圆心为蓝色方框“C1_o”，而窗口中所有点质心却是“C1_r”(小的蓝色圆圈)，很明显圆心和点的质心没有重合。所以移动圆心 C1_o 到质心 C1_r，这样我们就得到了一个新的窗口。这时又可以找到新窗口内所有点的质心，大多数情况下还是不重合的，所以重复上面的操作：将新窗口的中心移动到新的质心。就这样不停的迭代操作直到窗口的中心和其所包含点的质心重合为止（或者有一点小误差）。按照这样的操作我们的窗口最终会落在像素值（和）最大的地方。如上图所示“C2”是窗口的最后位址，我们可以看出来这个窗口中的像素点最多。整个过程如下图所示：

![image2](https://raw.githubusercontent.com/TonyStark1997/OpenCV-Python/master/6.Video%20Analysis%20/Image/image2.gif)

所以我们通常会通过直方图反投影图像和初始目标位置。当物体移动时，显然移动反映在直方图反投影图像中。因此，meanshift算法将窗口移动到具有最大密度的新位置。

### 2、OpenCV 中的 Meanshift

要在 OpenCV 中使用 Meanshift 算法首先我们要对目标对象进行设置，计算目标对象的直方图，这样在执行 meanshift 算法时我们就可以将目标对象反向投影到每一帧中去了。另外我们还需要提供窗口的起始位置。在这里我们值计算 H（Hue）通道的直方图，同样为了避免低亮度造成的影响，我们使用函数 cv2.inRange()将低亮度的值忽略掉。

```python
import numpy as np
import cv2 as cv

cap = cv.VideoCapture('slow.flv')

# take first frame of the video
ret,frame = cap.read()

# setup initial location of window
r,h,c,w = 250,90,400,125  # simply hardcoded the values
track_window = (c,r,w,h)

# set up the ROI for tracking
roi = frame[r:r+h, c:c+w]
hsv_roi =  cv.cvtColor(roi, cv.COLOR_BGR2HSV)
mask = cv.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
roi_hist = cv.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv.normalize(roi_hist,roi_hist,0,255,cv.NORM_MINMAX)

# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1 )

while(1):
    ret ,frame = cap.read()
    if ret == True:
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        dst = cv.calcBackProject([hsv],[0],roi_hist,[0,180],1)
        
        # apply meanshift to get the new location
        ret, track_window = cv.meanShift(dst, track_window, term_crit)
        
        # Draw it on image
        x,y,w,h = track_window
        img2 = cv.rectangle(frame, (x,y), (x+w,y+h), 255,2)
        cv.imshow('img2',img2)
        
        k = cv.waitKey(60) & 0xff
        if k == 27:
            break
        else:
            cv.imwrite(chr(k)+".jpg",img2)
            
    else:
        break
        
cv.destroyAllWindows()
cap.release()
```

我使用的视频中的三个帧如下：

![image3](https://raw.githubusercontent.com/TonyStark1997/OpenCV-Python/master/6.Video%20Analysis%20/Image/image3.jpg)

### 3、Camshift

你认真看上面的结果了吗？这里面还有一个问题。我们的窗口的大小是固定的，而汽车由远及近（在视觉上）是一个逐渐变大的过程，固定的窗口是不合适的。所以我们需要根据目标的大小和角度来对窗口的大小和角度进行修订。OpenCVLabs 为我们带来的解决方案（1988 年）：一个被叫做 CAMshift 的算法（连续自适应手段移位），由Gary Bradsky于1988年发表在他的论文“用于感知用户界面的计算机视觉面部跟踪”中。

Camshift算法首先应用meanshift。一旦meanshift收敛，它就会更新窗口的大小，$s = 2 \times \sqrt{\frac{M_{00}}{256}}$。它还计算最佳拟合椭圆的方向。同样，它将新的缩放搜索窗口和先前的窗口位置应用于meanshift。继续该过程直到满足所需的准确度。

![image4](https://raw.githubusercontent.com/TonyStark1997/OpenCV-Python/master/6.Video%20Analysis%20/Image/image4.gif)

### 4、OpenCV 中的 Camshift

它与meanshift几乎相同，但它返回一个旋转的矩形（这是我们的结果）和box参数（用于在下一次迭代中作为搜索窗口传递）。请参阅以下代码：

```python
import numpy as np
import cv2 as cv

cap = cv.VideoCapture('slow.flv')

# take first frame of the video
ret,frame = cap.read()

# setup initial location of window
r,h,c,w = 250,90,400,125  # simply hardcoded the values
track_window = (c,r,w,h)

# set up the ROI for tracking
roi = frame[r:r+h, c:c+w]
hsv_roi =  cv.cvtColor(roi, cv.COLOR_BGR2HSV)
mask = cv.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
roi_hist = cv.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv.normalize(roi_hist,roi_hist,0,255,cv.NORM_MINMAX)

# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1 )

while(1):
    ret ,frame = cap.read()
    if ret == True:
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        dst = cv.calcBackProject([hsv],[0],roi_hist,[0,180],1)
        
        # apply meanshift to get the new location
        ret, track_window = cv.CamShift(dst, track_window, term_crit)
        
        # Draw it on image
        pts = cv.boxPoints(ret)
        pts = np.int0(pts)
        img2 = cv.polylines(frame,[pts],True, 255,2)
        cv.imshow('img2',img2)
        
        k = cv.waitKey(60) & 0xff
        if k == 27:
            break
        else:
            cv.imwrite(chr(k)+".jpg",img2)
            
    else:
        break
        
cv.destroyAllWindows()
cap.release()
```

结果的三个框架如下所示：

![image5](https://raw.githubusercontent.com/TonyStark1997/OpenCV-Python/master/6.Video%20Analysis%20/Image/image5.jpg)

## 二、光流

***

### 目标：

本章节你需要学习以下内容:

    *我们将了解光流的概念及 Lucas-Kanade 光流法。
    *我们将使用cv.calcOpticalFlowPyrLK()等函数来跟踪视频中的特征点。

### 1、光流

由于目标对象或者摄像机的移动造成的图像对象在连续两帧图像中的移动被称为光流。它是一个 2D 向量场，可以用来显示一个点从第一帧图像到第二帧图像之间的移动。如下图所示（维基百科关于光流的文章）。

![image6](https://raw.githubusercontent.com/TonyStark1997/OpenCV-Python/master/6.Video%20Analysis%20/Image/image6.jpg)

上图显示了一个点在连续的五帧图像间的移动。箭头表示光流场向量。光流在很多领域中都很有用：

* 由运动重建结构
* 视频压缩
* 视频防抖等等

光流是基于以下假设下工作的：

1. 在连续的两帧图像之间（目标对象的）像素的灰度值不改变。
2. 相邻像素具有相似的运动。

考虑第一帧中的像素$I(x,y,t)$，它在dt时间之后移动距离$(dx,dy)$。根据第一条假设：灰度值不变。所以我们可以得到：

$$I(x,y,t) = I(x+dx, y+dy, t+dt)$$

然后对等号右侧采用泰勒级数展开，删除相同项并两边除以dt得到以下等式：

$$f_x u + f_y v + f_t = 0 \;$$

其中：

$$f_x = \frac{\partial f}{\partial x} \; ; \; f_y = \frac{\partial f}{\partial y}$$

$$u = \frac{dx}{dt} \; ; \; v = \frac{dy}{dt}$$

上边的等式叫做光流方程。其中 $f_x$ 和 $f_y$ 是图像梯度，同样 $f_t$ 是时间方向的梯度。但（u，v）是不知道的。我们不能在一个等式中求解两个未知数。有几个方法可以帮我们解决这个问题，其中的一个是 Lucas-Kanade 法。

### 2、Lucas-Kanade方法

现在我们要使用第二条假设，邻域内的所有点都有相似的运动。LucasKanade 法就是利用一个 3x3 邻域中的 9 个点具有相同运动的这一点。这样我们就可以找到$(f_x, f_y, f_t)$这 9 个点的光流方程，用它们组成一个具有两个未知数 9 个等式的方程组，这是一个约束条件过多的方程组。一个好的解决方法就是使用最小二乘拟合。下面就是求解结果：

$$\begin{bmatrix} u \\ v \end{bmatrix} = \begin{bmatrix} \sum_{i}{f_{x_i}}^2 & \sum_{i}{f_{x_i} f_{y_i} } \\ \sum_{i}{f_{x_i} f_{y_i}} & \sum_{i}{f_{y_i}}^2 \end{bmatrix}^{-1} \begin{bmatrix} - \sum_{i}{f_{x_i} f_{t_i}} \\ - \sum_{i}{f_{y_i} f_{t_i}} \end{bmatrix}$$

（你会发现上边的逆矩阵与 Harris 角点检测器非常相似，这说明角点很适合被用来做跟踪）

从使用者的角度来看，想法很简单，我们取跟踪一些点，然后我们就会获得这些点的光流向量。但是还有一些问题。直到现在我们处理的都是很小的运动。如果有大的运动怎么办呢？图像金字塔。我们可以使用图像金字塔的顶层，此时小的运动被移除，大的运动装换成了小的运动，现在再使用 Lucas-Kanade算法，我们就会得到尺度空间上的光流。

### 3、Lucas-KanadeOpenCV中的Lucas-Kanade光流

O上述所有过程都被 OpenCV 打包成了一个函数：cv2.calcOpticalFlowPyrLK()。现在我们使用这个函数创建一个小程序来跟踪视频中的一些点。要跟踪那些点呢？我们使用函数 cv2.goodFeatureToTrack() 来确定要跟踪的点。我们首先在视频的第一帧图像中检测一些 Shi-Tomasi 角点，然后我们使用 LucasKanade 算法迭代跟踪这些角点。我们要给函数 cv2.calcOpticlaFlowPyrLK()传入前一帧图像和其中的点，以及下一帧图像。函数将返回带有状态数的点，如果状态数是 1，那说明在下一帧图像中找到了这个点（上一帧中角点），如果状态数是 0，就说明没有在下一帧图像中找到这个点。我们再把这些点作为参数传给函数，如此迭代下去实现跟踪。代码如下：

```python
import numpy as np
import cv2 as cv

cap = cv.VideoCapture('slow.flv')

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0,255,(100,3))

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

while(1):
    ret,frame = cap.read()
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    # calculate optical flow
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    
    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]
    
    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        frame = cv.circle(frame,(a,b),5,color[i].tolist(),-1)
    img = cv.add(frame,mask)
    
    cv.imshow('frame',img)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
        
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)
    
cv.destroyAllWindows()
cap.release()
```

（(上面的代码没有对返回角点的正确性进行检查。图像中的一些特征点甚至在丢失以后，光流还会找到一个预期相似的点。所以为了实现稳定的跟踪，我们应该每个一定间隔就要进行一次角点检测。OpenCV 的官方示例中带有这样一个例子，它是每 5 帧进行一个特征点检测。它还对光流点使用反向检测来选取好的点进行跟踪。示例为/samples/python2/lk_track.py)

结果如下图所示：

![image7](https://raw.githubusercontent.com/TonyStark1997/OpenCV-Python/master/6.Video%20Analysis%20/Image/image7.jpg)

### 4、OpenCV中的密集光流

Lucas-Kanade 法是计算一些特征点的光流（我们上面的例子使用的是Shi-Tomasi 算法检测到的角点）。OpenCV 还提供了一种计算稠密光流的方法。它会图像中的所有点的光流。这是基于 Gunner_Farneback 的算法（2003 年）。

下面的例子就是使用上面的算法计算稠密光流。结果是一个带有光流向量（u，v）的双通道数组。通过计算我们能得到光流的大小和方向。我们使用颜色对结果进行编码以便于更好的观察。方向对应于 H（Hue）通道，大小对应于 V（Value）通道。代码如下：

```python
import cv2 as cv
import numpy as np
cap = cv.VideoCapture("vtest.avi")

ret, frame1 = cap.read()
prvs = cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255

while(1):
    ret, frame2 = cap.read()
    next = cv.cvtColor(frame2,cv.COLOR_BGR2GRAY)
    
    flow = cv.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)
    
    cv.imshow('frame2',bgr)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv.imwrite('opticalfb.png',frame2)
        cv.imwrite('opticalhsv.png',bgr)
    prvs = next
    
cap.release()
cv.destroyAllWindows()
```

请看下面的结果：

![image8](https://raw.githubusercontent.com/TonyStark1997/OpenCV-Python/master/6.Video%20Analysis%20/Image/image8.jpg)

OpenCV 的官方示例中有一个更高级的稠密光流算法，具体请参阅/samples/python2/opt_flow.py。

## 三、背景消除

***

### 目标：

本章节你需要学习以下内容:

    *我们将熟悉OpenCV中可用的背景消除方法。

### 1、基础

在很多基础应用中背景检出都是一个非常重要的步骤。例如顾客统计，使用一个静态摄像头来记录进入和离开房间的人数，或者是交通摄像头，需要提取交通工具的信息等。在所有的这些例子中，首先要将人或车单独提取出来。技术上来说，我们需要从静止的背景中提取移动的前景。

如果你有一张背景（仅有背景不含前景）图像，比如没有顾客的房间，没有交通工具的道路等，那就好办了。我们只需要在新的图像中减去背景就可以得到前景对象了。但是在大多数情况下，我们没有这样的（背景）图像，所以我们需要从我们有的图像中提取背景。如果图像中的交通工具还有影子的话，那这个工作就更难了，因为影子也在移动，仅仅使用减法会把影子也当成前景。真是一件很复杂的事情。

为了实现这个目的科学家们已经提出了几种算法。OpenCV 中已经包含了其中三种比较容易使用的方法。我们将逐一学习到它们。

#### （1）BackgroundSubtractorMOG

这是一个以混合高斯模型为基础的前景/背景分割算法。它是 P.KadewTraKuPong和 R.Bowden 在 2001 年提出的。它使用 K（K=3 或 5）个高斯分布混合对背景像素进行建模。使用这些颜色（在整个视频中）存在时间的长短作为混合的权重。背景的颜色一般持续的时间最长，而且更加静止。一个像素怎么会有分布呢？在 x，y 平面上一个像素就是一个像素没有分布，但是我们现在讲的背景建模是基于时间序列的，因此每一个像素点所在的位置在整个时间序列中就会有很多值，从而构成一个分布。

在编写代码时，我们需要使用函数：cv2.createBackgroundSubtractorMOG()创建一个背景对象。这个函数有些可选参数，比如要进行建模场景的时间长度，高斯混合成分的数量，阈值等。将他们全部设置为默认值。然后在整个视频中我们是需要使用 backgroundsubtractor.apply() 就可以得到前景的掩模了。

下面是一个简单的例子：

```python
import numpy as np
import cv2 as cv

cap = cv.VideoCapture('vtest.avi')
fgbg = cv.bgsegm.createBackgroundSubtractorMOG()

while(1):
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    cv.imshow('frame',fgmask)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
        
cap.release()
cv.destroyAllWindows()
```

（所有结果都显示在最后以供比较）。

#### （2）BackgroundSubtractorMOG2

这个也是以高斯混合模型为基础的背景/前景分割算法。它是以 2004 年和 2006 年 Z.Zivkovic 的两篇文章为基础的。这个算法的一个特点是它为每一个像素选择一个合适数目的高斯分布。（上一个方法中我们使用是 K 高斯分布）。这样就会对由于亮度等发生变化引起的场景变化产生更好的适应。

和前面一样我们需要创建一个背景对象。但在这里我们我们可以选择是否检测阴影。如果 detectShadows = True（默认值），它就会检测并将影子标记出来，但是这样做会降低处理速度。影子会被标记为灰色。

```python
import numpy as np
import cv2 as cv

cap = cv.VideoCapture('vtest.avi')

fgbg = cv.createBackgroundSubtractorMOG2()

while(1):
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    cv.imshow('frame',fgmask)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
        
cap.release()
cv.destroyAllWindows()
```

（最后给出的结果）

#### （3）BackgroundSubtractorGMG

此算法结合了静态背景图像估计和每个像素的贝叶斯分割。这是 2012 年Andrew_B.Godbehere，Akihiro_Matsukawa 和 Ken_Goldberg 在文章中提出的。

它使用前面很少的图像（默认为前 120 帧）进行背景建模。使用了概率前景估计算法（使用贝叶斯估计鉴定前景）。这是一种自适应的估计，新观察到的对象比旧的对象具有更高的权重，从而对光照变化产生适应。一些形态学操作如开运算闭运算等被用来除去不需要的噪音。在前几帧图像中你会得到一个黑色窗口。

对结果进行形态学开运算对与去除噪声很有帮助。

```python
import numpy as np
import cv2 as cv

cap = cv.VideoCapture('vtest.avi')

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
fgbg = cv.bgsegm.createBackgroundSubtractorGMG()

while(1):
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel)
    cv.imshow('frame',fgmask)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
        
cap.release()
cv.destroyAllWindows()
```

### 2、结果

#### （1）原始框架

下图显示了视频的第200帧

![image9](https://raw.githubusercontent.com/TonyStark1997/OpenCV-Python/master/6.Video%20Analysis%20/Image/image9.jpg)

#### （2）BackgroundSubtractorMOG的结果

![image10](https://raw.githubusercontent.com/TonyStark1997/OpenCV-Python/master/6.Video%20Analysis%20/Image/image10.jpg)

#### （3）BackgroundSubtractorMOG2的结果

灰色区域显示阴影区域。

![image11](https://raw.githubusercontent.com/TonyStark1997/OpenCV-Python/master/6.Video%20Analysis%20/Image/image11.jpg)

#### （4）BackgroundSubtractorGMG的结果

通过形态开口消除噪音。

![image12](https://raw.githubusercontent.com/TonyStark1997/OpenCV-Python/master/6.Video%20Analysis%20/Image/image12.jpg)