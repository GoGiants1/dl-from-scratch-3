import math

pil_available = True
try:
    from PIL import Image
except:
    pil_available = False
import numpy as np
from dezero import cuda

# 미니 배치를 만들어주는 DataLoader 클래스


class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=True, gpu=False):
        self.dataset = dataset  # 앞서 정의한 Dataset 인스턴스
        self.batch_size = batch_size  # 미니 배치 크기
        self.shuffle = shuffle  # 에포크별로 데이터를 섞을지 여부
        self.data_size = len(dataset)  # 데이터셋의 크기
        # 에포크당 최대 반복 수 (iter 객체 때문에 필요)
        self.max_iter = math.ceil(self.data_size / batch_size)
        self.gpu = gpu

        self.reset()

    def reset(self):
        self.iteration = 0
        if self.shuffle:
            self.index = np.random.permutation(len(self.dataset))
        else:
            self.index = np.arange(len(self.dataset))

    def __iter__(self):
        return self

    def __next__(self):
        if self.iteration >= self.max_iter:
            self.reset()
            raise StopIteration

        i, batch_size = self.iteration, self.batch_size
        batch_index = self.index[i * batch_size : (i + 1) * batch_size]
        batch = [self.dataset[i] for i in batch_index]

        xp = cuda.cupy if self.gpu else np
        # x = xp.array([example[0] for example in batch])
        # t = xp.array([example[1] for example in batch])
        x, t = zip(*batch)
        x = xp.array(x)
        t = xp.array(t)
        self.iteration += 1
        return x, t

    def next(self):
        return self.__next__()

    def to_cpu(self):
        self.gpu = False

    def to_gpu(self):
        self.gpu = True


class SeqDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, gpu=False):
        super().__init__(dataset=dataset, batch_size=batch_size, shuffle=False, gpu=gpu)

    def __next__(self):
        if self.iteration >= self.max_iter:
            self.reset()
            raise StopIteration

        jump = self.data_size // self.batch_size
        batch_index = [
            (i * jump + self.iteration) % self.data_size for i in range(self.batch_size)
        ]
        batch = [self.dataset[i] for i in batch_index]

        xp = cuda.cupy if self.gpu else np
        x = xp.array([example[0] for example in batch])
        t = xp.array([example[1] for example in batch])

        self.iteration += 1
        return x, t
