import numpy as np


class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None  # gradient calculated by backward method
        self.creator = None  # Parent(function) of this variable (링크드리스트 형태로 연결됨)

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        f = self.creator  # 1. Get a function
        if f is not None:
            # 2. Get the function's input (부모의 입력 변수, 이 곳에서는 `downstream gradient` 저장함)
            x = f.input
            x.grad = f.backward(self.grad)  # 3. Call the function's backward method
            x.backward()  # 4. 연결된 down stream을 따라 backward 호출(recursion과 유사한 흐름으로.)

            # 변수.backward -> 함수.backward -> 변수.backward -> 함수.backward -> ... -> 변수.backward (학습 parameter 까지)


class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        output.set_creator(self)  # Set parent(function)
        self.input = input
        self.output = output  # Set output
        return output

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()


class Square(Function):
    def forward(self, x):
        y = x**2
        return y

    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx


class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y

    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx


A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

# backward
y.grad = np.array(1.0)
y.backward()
print(x.grad)
