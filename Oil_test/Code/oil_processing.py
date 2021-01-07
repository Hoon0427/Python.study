import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings(action='ignore')


def image_histo(image_file):
    image = cv2.imread(image_file)
    orig = image.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edged = cv2.Canny(blur, 70, 150)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    
    (cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    cnts = sorted(cnts, key = cv2.contourArea, reverse=True)[:3]

    for c in cnts:
        peri = cv2.arcLength(c,True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            screenCnt = approx
            break

    cv2.drawContours(image,[screenCnt], -1, (0,255,0), 2)

#     rect = order_points(screenCnt.reshape(4,2) / r)
    rect = order_points(screenCnt.reshape(4,2))

    (topLeft, topRight, bottomRight, bottomLeft) = rect

    w1 = abs(bottomRight[0] - bottomLeft[0])
    w2 = abs(topRight[0] - topLeft[0])
    h1 = abs(topRight[1] - bottomRight[1])
    h2 = abs(topLeft[1] - bottomLeft[1])

    maxWidth = max([w1, w2])
    maxHeight = max([h1, h2])

    dst = np.float32([[0,0], [maxWidth-1,0], [maxWidth-1, maxHeight-1], [0,maxHeight-1]])

    M = cv2.getPerspectiveTransform(rect, dst)

    warped = cv2.warpPerspective(orig,M,(maxWidth,maxHeight))

    height, width = warped.shape[:2]

    if height < width:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    
    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    warped_01 = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY)
    df_warped = pd.DataFrame(warped_01)
    
    df_warped_len = df_warped.max().max() - df_warped.min().min()
    
    plt.figure(figsize=(15,10))

    sns.distplot(df_warped, bins= df_warped_len, hist=True, kde=False, rug=False)
    
    plt.show()
    
    
def image_processing(image_file, img=False, heatmap=False, hist=False):
    image = cv2.imread(image_file)
    orig = image.copy()

################ cv2으로 기름종이 이미지 추출  ################

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edged = cv2.Canny(blur, 70, 150)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    
    (cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    cnts = sorted(cnts, key = cv2.contourArea, reverse=True)[:3]

    for c in cnts:
        peri = cv2.arcLength(c,True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            screenCnt = approx
            break

    cv2.drawContours(image,[screenCnt], -1, (0,255,0), 2)

    
## 직사각형으로 좌표 배치

    def order_points(pts):
        rect = np.zeros((4,2), dtype = "float32")

        s = pts.sum(axis = 1)

        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis = 1)

        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        return rect


    rect = order_points(screenCnt.reshape(4,2))

    (topLeft, topRight, bottomRight, bottomLeft) = rect

    w1 = abs(bottomRight[0] - bottomLeft[0])
    w2 = abs(topRight[0] - topLeft[0])
    h1 = abs(topRight[1] - bottomRight[1])
    h2 = abs(topLeft[1] - bottomLeft[1])

    maxWidth = max([w1, w2])
    maxHeight = max([h1, h2])

    dst = np.float32([[0,0], [maxWidth-1,0], [maxWidth-1, maxHeight-1], [0,maxHeight-1]])

    M = cv2.getPerspectiveTransform(rect, dst)

    warped = cv2.warpPerspective(orig,M,(maxWidth,maxHeight))

    height, width = warped.shape[:2]

    
## 추출한 이미지가 가로일 때 세로로 회전
    if height < width:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    warped_01 = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY)
    df_warped = pd.DataFrame(warped_01)


## 계산을 위한 histogram
    warped_bins = df_warped.max().max() - df_warped.min().min()
    warped_count = np.histogram(df_warped, bins=warped_bins)

    aaa = []
    bbb = []
    aaa.append(warped_count[0])
    bbb.append(warped_count[1])

    bbb[0] = bbb[0][1:]

    aaa = list(map(int, aaa[0]))
    bbb = list(map(int, bbb[0]))

# 최대값 이상의 값을 갖는 cell을 None으로 replace
    df_warped[df_warped > bbb[np.where(warped_count[0] == warped_count[0].max())[0][0]]] = None
    
################ 이미지 띄우기 ################
    if (img==True or heatmap==True or hist==True):
        
        count = 0
        if (img==True):
            count+=1
        if (heatmap==True):
            count+=1
        if (hist==True):
            count+=1
        
        if (count==1):

            if (img == True):
                plt.figure(figsize=(3,5))
                plt.xticks([])
                plt.yticks([])
                plt.imshow(warped, aspect='auto')

            if (heatmap == True):
                sns.heatmap(df_warped, cbar=False)
                plt.xticks([])
                plt.yticks([])
                plt.show()

            if (hist == True):
                sns.distplot(warped_01, bins= warped_bins, hist=True, kde=False, rug=False)
            
        else:
            fig, axes = plt.subplots(ncols=count, figsize=(15,10))

            graph_count = -1
            if (img == True):
                graph_count+=1
                axes[graph_count].imshow(warped, aspect='auto')

            if (heatmap == True):
                graph_count+=1
                sns.heatmap(df_warped, cbar=False, ax=axes[graph_count])

            if (hist == True):
                graph_count+=1
                sns.distplot(warped_01, bins= warped_bins, hist=True, kde=False, rug=False, ax=axes[graph_count])
                axes[graph_count].grid(True)
            plt.show()
        
################ 픽셀값 60까지의 픽셀 빈도수 더하기  ################
    sebum = sum(aaa[:bbb.index(60)+1])

# 유분량
    print('유분량: ', 100*round((sebum/sum(aaa[:aaa.index(max(aaa))+1])),3),'%')