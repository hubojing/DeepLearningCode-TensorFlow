#在jupyter中使用matplotlib显示图像需设为inline模式，否则不会显示图像
%matplotlib inline

import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

#设置随机数种子
np.random.seed(5)

#直接采用np生成等差数列的方法，生成100个点，每个点的取值在-1~1之间
x_data = np.linspace(-1, 1, 100)

# y=2x+1+噪声，噪声的维度与x_data一致
y_data = 2*x_data + 1.0 + np.random.randn(*x_data.shape) * 0.4

#画出随机生成数据的散点图
#plt.scatter(x_data, y_data)

#画出线性函数y=2x+1
#plt.plot(x_data, 2*x_data + 1.0, color='red', linewidth=3)

x = tf.placeholder("float", name = "x")
y = tf.placeholder("float", name = "y")


def model(x, w, b):
	return tf.multiply(x, w) + b

#构建线性函数的斜率
w = tf.Variable(1.0, name="w0")

#构建线性函数的截距
b = tf.Variable(0.0, name="b0")

#pred是预测值，前向计算
pred = model(x, w, b)

#迭代次数（训练轮数）
train_epochs = 10

#学习率
learning_rate = 0.05

#控制显示loss值的粒度
display_step = 10

#采用均方差作为损失函数
loss_function = tf.reduce_mean(tf.square(y - pred))

#梯度下降优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_function)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

#开始训练，采用SGD随机梯度下降优化方法
step = 0#训练步数
loss_list = []

for epoch in range(train_epochs):
    for xs,ys in zip(x_data, y_data):
        _, loss = sess.run([optimizer,loss_function], feed_dict={x: xs, y: ys})
        #显示损失值loss,display_step控制报告的粒度
        #若display_step为2，则将每训练2个样本输出一次损失值
        loss_list.append(loss)
        step = step + 1
        if step % display_step == 0:
            print("Train Epoch:","%02d" % (epoch+1), "Step: %03d" % (step), "loss=", "{:.9f}".format(loss))
            
    b0temp = b.eval(session=sess)
    w0temp = w.eval(session=sess)
    #plt.plot(x_data, w0temp * x_data + b0temp)
    
print("w:", sess.run(w))
print("b:", sess.run(b))

#plt.scatter(x_data, y_data, label='Original data')
#plt.plot(x_data, x_data * sess.run(w) + sess.run(b), label='Fitted line', color='r', linewidth=3)
#plt.legend(loc=2)#通过参数loc指定图例位置

x_test = 3.21
predict = sess.run(w) * x_test + sess.run(b)
print("预测值：%f" % predict)

#plt.plot(loss_list)
plt.plot(loss_list,'r+')

[x for x in loss_list if x>1]
