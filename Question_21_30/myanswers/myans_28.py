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

    # x, y は範囲外にしてはダメー 負の数だめ, 129の範囲まで
    # これは本来入らない計算の部分にも代入されてしまうのでは？
    x = np.minimum(np.maximum(x, 0), Hor+1).astype(np.int) 
    y = np.minimum(np.maximum(y, 0), Ver+1).astype(np.int)

    result[New_y, New_x] = img[y, x]
    result = result[1:New_Hor, 1:New_Ver]
    result = result.astype(np.uint8)

    '''
    # for文 ver
    for x in range(Hor):
        for y in range(Ver):
            ix = int(a * x + b * y + tx)
            iy = int(c * x + d * y + ty)

            if (0 <= ix and ix < Hor) and (0 <= iy and iy < Ver):
                result[ix, iy] = img[x, y].copy()

    '''
    return result

img = cv2.imread("../imori.jpg")

result = Affin(img, a=1, b=0, c=0, d=1, tx=20, ty=10)

cv2.imshow("result", result)
cv2.imwrite("myans_28.jpg", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
