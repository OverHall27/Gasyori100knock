import cv2
import numpy as np

##並行移動したら，空白部分は0にしなきゃいけないよね

def Affin(_img, a, b, c, d, tx, ty):
    #高さ, 幅, 色
    Ver, Hor, Col = _img.shape

    # for文 ナシver
    # 拡大/縮小に合わせた処理
    New_Hor = np.round(Hor * a).astype(np.int)
    New_Ver = np.round(Ver * d).astype(np.int)
    result = np.zeros((New_Ver+1, New_Hor+1, Col), dtype=np.float)

    # 画像の周りにゼロpadding
    img = np.zeros((Ver + 2, Hor + 2, Col), dtype=np.float)
    img[1: Ver + 1, 1: Hor + 1] = _img
    # resultは1列1行多くつくる

    # resultの画座標列 これ２次元じゃなきゃだめ?
    # => １次元同士だと，対角成分しか計算されない
    New_x = np.tile(np.arange(New_Hor), (New_Ver, 1))
    New_y = np.arange(New_Ver).repeat(New_Hor).reshape(New_Ver, -1)

    #逆行列使って元画像の座標値求める, ゼロパディグでずれてる分 +1 ?
    adbc = a * d - b * c
    x = np.round((d * New_x - b * New_y) / adbc).astype(np.int) -  tx + 1
    y = np.round((a * New_y - c * New_x) / adbc).astype(np.int) -  ty + 1

    # 回転してるとこのまま代入するとずれてしまうかも, 
    # 回転後の画像の中心を考慮してずらすのが必要

    x += ((Hor/2) - ((x.max() + x.min()) / 2.)).astype(np.int)
    y += ((Ver/2) - ((y.max() + y.min()) / 2.)).astype(np.int)

    # x, y は範囲外にしてはダメー 負の数だめ, 129の範囲まで
    # これは本来入らない計算の部分にも代入されてしまうのでは？
    x = np.minimum(np.maximum(x, 0), Hor+1).astype(np.int) 
    y = np.minimum(np.maximum(y, 0), Ver+1).astype(np.int)

    # 回転してるとこのまま代入するとずれてしまうかも, 
    # 回転後の画像の中心を考慮してずらすのが必要

    result[New_y, New_x] = img[y, x]
    result = result[:New_Ver, :New_Hor]
    result = result.astype(np.uint8)
    
    return result

img = cv2.imread("../imori.jpg")
Ver, Hor, Col = img.shape
result_1= Affin(img, a=1, b=30/Hor, c=0, d=1, tx=0, ty=0)
result_2 = Affin(img, a=1, b=0, c=30/Ver, d=1, tx=0, ty=0)
result_3 = Affin(img, a=1, b=30/Hor, c=30/Ver, d=1, tx=0, ty=0)

cv2.imshow("result1", result_1)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow("result2", result_2)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow("result3", result_3)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("myans_31_1.jpg", result_1)
cv2.imwrite("myans_31_2.jpg", result_2)
cv2.imwrite("myans_31_3.jpg", result_3)
