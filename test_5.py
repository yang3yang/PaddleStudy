from __future__ import print_function  # 将python3中的print特性导入当前版本
import os
from PIL import Image  # 导入图像处理模块
import matplotlib.pyplot as plt
import numpy
import paddle  # 导入paddle模块
import paddle.fluid as fluid


def softmax_regression():
    """
        定义softmax分类器：
            一个以softmax为激活函数的全连接层
        Return:
            predict_image -- 分类的结果
    """
    img = fluid.data(name='img', shape=[None, 1, 28, 28], dtype='float32')
    predict = fluid.layers.fc(input=img, size=10, act='softmax')
    return predict


def multilayer_preceptron():
    """
    定义多层感知机分类器：
        含有两个隐藏层（全连接层）的多层感知器
        其中前两个隐藏层的激活函数采用 ReLU，输出层的激活函数用 Softmax

    Return:
        predict_image -- 分类的结果
    """
    img = fluid.data('img', shape=[None, 1, 28, 28], dtype='float32')
    hidden = fluid.layers.fc(input=img, size=200, act='relu')
    hidden = fluid.layers.fc(input=hidden, size=200, act='relu')
    prediction = fluid.layers.fc(input=hidden, size=10, act='softmax')
    return prediction


def convolutional_neural_network():
    """
    定义卷积神经网络分类器：
        输入的二维图像，经过两个卷积-池化层，使用以softmax为激活函数的全连接层作为输出层

    Return:
        predict -- 分类的结果
    """
    img = fluid.data(name='img', shape=[None, 1, 28, 28], dtype='float32')
    conv_pool_1 = fluid.nets.simple_img_conv_pool(input=img, filter_size=5, num_filters=20, pool_size=2, pool_stride=2,
                                                  act='relu')
    conv_pool_1 = fluid.layers.batch_norm(conv_pool_1)

    conv_pool_2 = fluid.nets.simple_img_conv_pool(input=conv_pool_1, filter_size=5, num_filters=50, pool_size=2,
                                                  pool_stride=2, act='relu')
    prediction = fluid.layers.fc(input=conv_pool_2, size=10, act='softmax')
    return prediction


def train_program():
    """
    配置train_program

    Return:
        predict -- 分类的结果
        avg_cost -- 平均损失
        acc -- 分类的准确率

    """
    label = fluid.data(name='label', shape=[None, 1], dtype='int64')
    # predict = softmax_regression()
    # predict = multilayer_preceptron()
    predict = convolutional_neural_network()

    cost = fluid.layers.cross_entropy(input=predict, label=label)

    avg_cost = fluid.layers.mean(cost)
    acc = fluid.layers.accuracy(input=predict, label=label)
    return predict, [avg_cost, acc]


def optimizer_program():
    return fluid.optimizer.Adam(learning_rate=0.001)


BATCH_SIZE = 64
train_reader = paddle.batch(paddle.reader.shuffle(paddle.dataset.mnist.train(), buf_size=500), batch_size=BATCH_SIZE)
test_reader = paddle.batch(paddle.dataset.mnist.test(), batch_size=BATCH_SIZE)


def event_handler(pass_id, batch_id, cost):
    print("Pass %d, Batch %d, Cost %f" % (pass_id, batch_id, cost))


from paddle.utils.plot import Ploter

train_prompt = "Train cost"
test_prompt = "Test cost"
cost_ploter = Ploter(train_prompt, test_prompt)


def event_handler_plot(ploter_title, step, cost):
    cost_ploter.append(ploter_title, step, cost)
    cost_ploter.plot()


use_cuda = False
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
exe = fluid.Executor(place)

prediction, [avg_loss, acc] = train_program()

feeder = fluid.DataFeeder(feed_list=['img', 'label'], place=place)

optimizer = optimizer_program()
optimizer.minimize(avg_loss)

PASS_NUM = 5
epochs = [epoch_id for epoch_id in range(PASS_NUM)]

save_dirname = 'd:/model_dir/recognize_digits.inference.model'


def train_test(train_test_program, train_test_feed, train_test_reader):
    acc_set = []
    avg_loss_set = []
    for test_data in train_test_reader():
        acc_np, avg_loss_np = exe.run(program=train_test_program, feed=train_test_feed.feed(test_data),
                                      fetch_list=[acc, avg_loss])
        acc_set.append(float(acc_np))
        avg_loss_set.append(float(avg_loss_np))

    acc_val_mean = numpy.array(acc_set).mean()
    avg_loss_val_mean = numpy.array(avg_loss_set).mean()

    return avg_loss_val_mean, acc_val_mean


exe.run(fluid.default_startup_program())

main_program = fluid.default_main_program()
test_program = fluid.default_main_program().clone(for_test=True)

lists = []
step = 0
for epoch_id in epochs:
    for step_id, data in enumerate(train_reader()):
        metrics = exe.run(main_program, feed=feeder.feed(data), fetch_list=[avg_loss, acc])
        if step % 100 == 0:
            print("Pass %d, Batch %d, Cost %f" % (step, epoch_id, metrics[0]))
            event_handler_plot(train_prompt, step, metrics[0])
        step += 1
    avg_loss_val, acc_val = train_test(train_test_program=test_program, train_test_reader=test_reader,
                                       train_test_feed=feeder)
    print("Test with Epoch %d, avg_cost:%s， acc：%s" % (epoch_id, avg_loss_val, acc_val))
    event_handler_plot(test_prompt, step, metrics[0])

    lists.append((epoch_id, avg_loss_val, acc_val))

    if save_dirname is not None:
        fluid.io.save_inference_model(save_dirname, ["img"], [prediction], exe, model_filename=None,
                                      params_filename=None)

best = sorted(lists, key=lambda list: float(list[1]))[0]
print('Best pass is %s, testing Avgcost is %s' % (best[0], best[1]))
print('The classification accuracy is %.2f%%' % (float(best[2]) * 100))


def load_image(file):
    im = Image.open(file).convert('L')
    im = im.resize((28, 28), Image.ANTIALIAS)
    im = numpy.array(im).reshape(1, 1, 28, 28).astype(numpy.float32)
    im = im / 255.0 * 2.0 - 1.0

    return im


cur_dir = os.getcwd()

inference_scope = fluid.core.Scope()
with fluid.scope_guard(inference_scope):
    [inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(save_dirname, exe, None, None)

    tensor_img = load_image(cur_dir + '/image/infer_3.png')
    results = exe.run(inference_program, feed={feed_target_names[0]: tensor_img}, fetch_list=fetch_targets)
    lab = numpy.argsort(results)
    print("Inference result of image/infer_3.png is:%d" % lab[0][0][-1])

    tensor_img = load_image(cur_dir + '/image/infer_6.jpg')
    results = exe.run(inference_program, feed={feed_target_names[0]: tensor_img}, fetch_list=fetch_targets)
    lab = numpy.argsort(results)
    print("Inference result of image/infer_6.jpg is:%d" % lab[0][0][-1])

    tensor_img = load_image(cur_dir + '/image/infer_9.jpg')
    results = exe.run(inference_program, feed={feed_target_names[0]: tensor_img}, fetch_list=fetch_targets)
    lab = numpy.argsort(results)
    print("Inference result of image/infer_9.jpg is:%d" % lab[0][0][-1])
