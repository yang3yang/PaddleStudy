import paddle.fluid as fluid
import paddle
import numpy as np
import os
import matplotlib.pyplot as plt

BUF_SIZE = 500
BATCH_SIZE = 20

train_reader = paddle.batch(paddle.reader.shuffle(paddle.dataset.uci_housing.train(), buf_size=BUF_SIZE),
                            batch_size=BATCH_SIZE)
test_reader = paddle.batch(paddle.reader.shuffle(paddle.dataset.uci_housing.test(), buf_size=BUF_SIZE),
                           batch_size=BATCH_SIZE)

train_data = paddle.dataset.uci_housing.train();
sampledata = next(train_data())
print(sampledata)

x = fluid.layers.data(name='x', shape=[13], dtype='float32')
y = fluid.layers.data(name='y', shape=[1], dtype='float32')
y_predict = fluid.layers.fc(input=x, size=1, act=None)

cost = fluid.layers.square_error_cost(input=y_predict, label=y)
avg_cost = fluid.layers.mean(cost)

optimizer = fluid.optimizer.SGDOptimizer(learning_rate=0.001)
opts = optimizer.minimize(avg_cost)
test_program = fluid.default_main_program().clone(for_test=True)

use_cuda = False
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

feeder = fluid.DataFeeder(place=place, feed_list=[x, y])

iter = 0
iters = []
train_costs = []


def draw_train_process(iter, train_costs):
    plt.figure(1)
    title = "training cost"
    plt.title(title, fontsize=24)
    plt.xlabel("iter", fontsize=14)
    plt.ylabel("cost", fontsize=14)
    plt.plot(iters, train_costs, color='red', label='training cost')
    plt.grid()
    plt.savefig('d:/test01.png')


EPOCH_NUM = 50
model_save_dir = 'd:/model_dir/fit_a_line.inference.model'

for pass_id in range(EPOCH_NUM):
    train_cost = 0
    for batch_id, data in enumerate(train_reader()):
        train_cost = exe.run(program=fluid.default_main_program(), feed=feeder.feed(data), fetch_list=[avg_cost])
        if batch_id % 40 == 0:
            print("Pass:%d, Cost:%0.5f" % (pass_id, train_cost[0][0]))  # 打印最后一个batch的损失值
        iter = iter + BATCH_SIZE
        iters.append(iter)
        train_costs.append(train_cost[0][0])

    test_cost = 0
    for batch_id, data in enumerate(test_reader()):
        test_cost = exe.run(program=test_program, feed=feeder.feed(data), fetch_list=[avg_cost])
    print('Test:%d, Cost:%0.5f' % (pass_id, test_cost[0][0]))

if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)
print('save models to %s' % model_save_dir)
fluid.io.save_inference_model(model_save_dir, ['x'], [y_predict], exe)
draw_train_process(iters, train_costs)

infer_exe = fluid.Executor(place)
inference_scope = fluid.core.Scope()

infer_results = []
groud_truths = []


# 绘制真实值和预测值对比图
def draw_infer_result(groud_truths, infer_results):
    plt.figure(2)
    title = 'Boston'
    plt.title(title, fontsize=24)
    x = np.arange(1, 20)
    y = x
    plt.plot(x, y)
    plt.xlabel('ground truth', fontsize=14)
    plt.ylabel('infer result', fontsize=14)
    plt.scatter(groud_truths, infer_results, color='green', label='training cost')
    plt.grid()
    plt.savefig('d:/test02.png')


with fluid.scope_guard(inference_scope):
    [inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(model_save_dir, infer_exe)

    infer_reader = paddle.batch(paddle.dataset.uci_housing.test(), batch_size=200)

    test_data = next(infer_reader())
    test_x = np.array([data[0] for data in test_data]).astype("float32")
    test_y = np.array([data[1] for data in test_data]).astype("float32")
    results = infer_exe.run(inference_program, feed={feed_target_names[0]: np.array(test_x)}, fetch_list=fetch_targets)

    print("infer results (House Price)")
    for idx, val in enumerate(results[0]):
        print("%d:%.2f" % (idx, val))
        infer_results.append(val)
    print("ground truth:")
    for idx, val in enumerate(test_y):
        print("%d:%.2f" % (idx, val))
        groud_truths.append(val)
    draw_infer_result(groud_truths, infer_results)
