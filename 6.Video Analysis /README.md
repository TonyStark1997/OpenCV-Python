# 第六章：视频分析

本章节你将学习Meanshift和Camshift、光流和背景减除等OpenCV视频分析的相关内容。

更多内容请关注我的GitHub库：https://github.com/TonyStark1997，如果喜欢，star并follow我！

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

    *我们将使用Lucas-Kanade方法理解光流的概念及其估计。
    *我们将使用cv.calcOpticalFlowPyrLK()等函数来跟踪视频中的特征点。

### 1、光流
