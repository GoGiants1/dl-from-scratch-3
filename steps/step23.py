# 패키지로 정리
# 모듈: 파이썬 파일. 다른 파이썬 파일에서 불러와 사용할 수 있는 파이썬 파일
# 패키지: 모듈을 모아놓은 것. 먼저 디렉토리를 만들고 그 안에 모듈을 추가.
# 라이브러리: 여러 패키지를 묶은 것. 하나 이상의 디렉토리로 구성.
# Add import path for the dezero directory.
if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from dezero import Variable


x = Variable(np.array(1.0))
y = (x + 3) ** 2
y.backward()

print(y)
print(x.grad)
