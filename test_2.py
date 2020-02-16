import paddle.fluid as fluid
import numpy as np

a = fluid.layers.create_tensor(dtype='int64', name='a')
b = fluid.layers.create_tensor(dtype='int64', name='b')

y = fluid.layers.sum(x=[a, b])

place = fluid.CPUPlace()
exe = fluid.executor.Executor(place)

exe.run(fluid.default_startup_program())

a1 = np.array([[3, 2, 1], [7, 8, 9]]).astype('int64')
b1 = np.array([[1, 1, 1], [1, 1, 1]]).astype('int64')

# 进行运算，并把y的结果输出
out_a, out_b, result = exe.run(program=fluid.default_main_program(),
                               feed={'a': a1, 'b': b1},
                               fetch_list=[a, b, y])
print(out_a, " + ", out_b, " = ", result)
