import weakref
import numpy as np
import contextlib


# Step 18: 메모리 절약 모드 구현하기
# 1. 역전파시 불필요한 미분값을 메모리에서 삭제하여 메모리 사용량을 줄이는 기능을 구현한다. -> 중간에 있는 변수들의 grad 값 제거.
# 2. 역전파가 필요 없을 때, 불필요한 계산 생략


class Config:
    enable_backprop = True


@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    # Config 클래스의 name 속성을 value로 덮어씀.
    setattr(Config, name, value)
    try:
        yield
    finally:
        # 원래 값 복원
        setattr(Config, name, old_value)


def no_grad():
    return using_config("enable_backprop", False)


class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError("{} is not supported".format(type(data)))

        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def cleargrad(self):
        self.grad = None

    def backward(self, retain_grad=False):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs: list["Function"] = []
        seen_set = set()

        def add_func(f: "Function"):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            gys = [output().grad for output in f.outputs]  # output is weakref
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    add_func(x.creator)

            if not retain_grad:
                for y in f.outputs:
                    y().grad = None
                    # y is weakref -> y() is Variable.
                    # x <- f <- y 순으로 역전파가 일어난 뒤, y의 grad를 None으로 설정하여 메모리에서 삭제한다.


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        # Backpropagation 하는 경우만 function과 variable을 연결해준다.
        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs):
        raise NotImplementedError()

    def backward(self, gys):
        raise NotImplementedError()


class Square(Function):
    def forward(self, x):
        y = x**2
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx


def square(x):
    return Square()(x)


class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y

    def backward(self, gy):
        return gy, gy


def add(x0, x1):
    return Add()(x0, x1)


x0 = Variable(np.array(1.0))
x1 = Variable(np.array(1.0))
t = add(x0, x1)
y = add(x0, t)
y.backward()
print(y.grad, t.grad)  # None None
print(x0.grad, x1.grad)  # 2.0 1.0


with using_config("enable_backprop", False):
    x = Variable(np.array(2.0))
    y = square(x)

with no_grad():
    x = Variable(np.array(2.0))
    y = square(x)
