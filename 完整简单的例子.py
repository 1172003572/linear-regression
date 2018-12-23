
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt #使图形可视化

#定义一个添加层   activation_function=None  表示一个线性的方程 inputs表示传入的数据  in_size 表示传入的大小 out_size表示传出的大小
def add_layer(inputs, in_size, out_size, activation_function=None):
    #权重定义为一个数组   一般为随机变量
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    #推荐初始值不为零，所以加0.1
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    #预测出来的值  换没有被激活
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    #开始激活
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

#加维度 在-1到1之间有300个单位 [:,np.newaxis]表示增加一个维度
x_data = np.linspace(-1,1,300)[:, np.newaxis]
#加一个照点 不是跟据 二次函数分布   而是在函数线两边分布 x_data.shape表示和x_data的格式一样
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise
# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])
#第一层隐藏层 xs表示输入 1表示输入只有一个神经元 10表示隐藏层有十个神经元   最后一个参数表示激励方程
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
# 输出层 l1输入  10隐藏层神经元 1输出层神经元
prediction = add_layer(l1, 10, 1, activation_function=None)
#reduction_indices表示处理维度的问题
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                     reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# important step
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

fig=plt.figure()#生成一个图片框
ax=fig.add_subplot(1,1,1) #使图可以连续性
ax.scatter(x_data,y_data) #以点的形式进行输出
plt.ion()
plt.show()


for i in range(1000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if i % 50== 0:
        try:
            ax.lines.remove(lines[0])
            #摸除掉上一条线
        except Exception:
            pass
      #  print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
        prediction_value=sess.run(prediction,feed_dict={xs:x_data})# 返回一个prediction  表示预测数据的y轴
        lines=ax.plot(x_data, prediction_value, 'r-', lw=5) #表示将这条线画出来
       
        plt.pause(1)#暂停0.1秒 在输出