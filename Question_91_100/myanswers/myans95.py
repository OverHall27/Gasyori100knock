import numpy as np

np.random.seed(0)

class NN:
    # constracter, __init__の引数はクラス内変数だったり？
    # 予めどの変数が定義されてないといけないということはないらしい

    # あと中間層を一つ増やすので，wは増える
    def __init__(self, ind=2, w1=64, w2=64, outd=1, lr=0.1):
        # np.random normal は正規分布乱数
        # ランダムでWeightとBを設定して後で修正してゆく
        # np.random.normal(average, sigma, paxis0, axis1, ...])
        self.w1 = np.random.normal(0, 1, [ind, w1])
        self.b1 = np.random.normal(0, 1, [w1])

        self.w2 = np.random.normal(0, 1, [w1, w2])
        self.b2 = np.random.normal(0, 1, [w2])

        self.wout = np.random.normal(0, 1, [w2, outd])
        self.bout = np.random.normal(0, 1, [outd])
        self.lr = lr

    def forward(self, x):
        # x is input
        # dot -> 1次元同士は内積, ２次元配列は行列積, @でも同じように行列積ができる
        # 今回sigmoid関数を使っているが，他にもいろいろな種類がある

        self.z1 = x
        # 第１層
        self.z2 = sigmoid(np.dot(self.z1, self.w1) + self.b1)
        # １層増やした中間層.
        self.z3 = sigmoid(np.dot(self.z2, self.w2) + self.b2)
        # 出力はこれ
        self.out = sigmoid(np.dot(self.z3, self.wout) + self.bout)

        return self.out

    # 当然どれくらい確からしい予測をできたのが評価する必要がある．損失関数にもいろいろあるが，
    # 今回はsum of sqaures をつかう．いわゆる誤差２乗和
    # 損失関数を最も少なくなるようにするのが目標になる

    # 誤差伝搬を計算して，重みとbの値をupdateしなければいけない．これがbackpropagation
    # しかし，最低のyにどのw, b が対応しているか，直接逆算はできない．
    # しかし，誤差二乗和をwで工夫して偏微分すると，wの値が求められるようになる．
    # そのためには, z = Wx + b となるzを利用する

    def train(self, x, t):
        # backpropagation output layer
        #En = t * np.log(self.out) + (1-t) * np.log(1-self.out)

        # サイト式
        # d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        # d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))
        # maybe ... sigmoid_derivative = z(1-z), z -> 各層の出力に相当
        # ndarrayは array.Tで転置行列が求められる！！

        #En = (t - self.out) * self.out * (1 - self.out)
        En = 2*(t - self.out) * self.out * (1 - self.out)
        #np.array([En for _ in range(t.shape[0])])
        grad_En = En 

        grad_wout = np.dot(self.z3.T, En)
        grad_bout = np.dot(np.ones([En.shape[0]]), En)

        # 学習係数をかけて加算
        self.wout += self.lr * grad_wout
        #np.expand_dims(grad_wout, axis=-1)
        self.bout += self.lr * grad_bout

        ## ↑↑ ここまではOK ↑↑

        # backpropagation inter layer
        grad_u2 = np.dot(En, self.wout.T) * self.z3 * (1 - self.z3)
        # 上式はサイト式のこの部分に当たる 
        # np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)

        grad_w2 = np.dot(self.z2.T, grad_u2)
        # grad_us -> (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))
        # grad_u2 = np.dot(En, self.wout.T) * self.z3 * (1 - self.z3)
 
        grad_b2 = np.dot(np.ones([grad_u2.shape[0]]), grad_u2)
        self.w2 += self.lr * grad_w2
        self.b2 += self.lr * grad_b2

        grad_u1 = np.dot(grad_u2, self.w2.T) * self.z2 * (1 - self.z2)

        grad_w1 = np.dot(self.z1.T, grad_u1)
        grad_b1 = np.dot(np.ones([grad_u1.shape[0]]), grad_u1)
        self.w1 += self.lr * grad_w1
        self.b1 += self.lr * grad_b1

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

# 排他的論理和？の学習？
train_x = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
train_t = np.array([[0], [1], [1], [0]], dtype=np.float32)

nn = NN(ind=train_x.shape[1])

# train
for i in range(1000):
    # WeightとBiasesを元に出力層yを計算
    nn.forward(train_x)
    #　計算したyを元にWeight, Biasesを修正
    nn.train(train_x, train_t)
    # 上記ふたつを何回も繰り返して学習すんぞ

# test
for j in range(4):
    x = train_x[j]
    t = train_t[j]
    print("in:", x, "pred:", nn.forward(x))
    

