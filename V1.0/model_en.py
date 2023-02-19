import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.python.keras.layers import Conv2D, Activation, MaxPool2D, Dropout, Flatten, Dense, AvgPool2D, Input
from tensorflow.python.keras import Model
from tensorflow.python.layers.normalization import BatchNormalization
import cv2 as cv

np.set_printoptions(threshold=np.inf)

TRAIN_DIR = "data/enu_train"
TEST_DIR = "data/enu_test"
checkpoint_save_path = "./checkpoint_en/ResNet18.ckpt"
save_path = './weights_en/'
# 英文图片重置的宽、高
IMAGE_WIDTH = 20
IMAGE_HEIGHT = 20
CLASSIFICATION_COUNT = 34
LABEL_DICT = {
    '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
    'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15, 'G': 16, 'H': 17, 'J': 18, 'K': 19,
    'L': 20, 'M': 21, 'N': 22, 'P': 23, 'Q': 24, 'R': 25, 'S': 26, 'T': 27, 'U': 28, 'V': 29,
    'W': 30, 'X': 31, 'Y': 32, 'Z': 33
}

# 设置GPU内存为陆续分配，防止一次性的全部分配GPU内存，导致系统过载
physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass


def load_data(dir_path):
    data = []
    labels = []

    for item in os.listdir(dir_path):
        item_path = os.path.join(dir_path, item)
        if os.path.isdir(item_path):
            for subitem in os.listdir(item_path):
                subitem_path = os.path.join(item_path, subitem)
                gray_image = cv.imread(subitem_path, cv.IMREAD_GRAYSCALE)
                resized_image = cv.resize(gray_image, (IMAGE_WIDTH, IMAGE_HEIGHT))
                data.append(resized_image.ravel())
                labels.append(LABEL_DICT[item])

    return np.array(data), np.array(labels)


# 本质：完成数据的正则化
def normalize_data(data):
    return (data - data.mean()) / data.max()


# 构建 独热编码
def onehot_labels(labels):
    onehots = np.zeros((len(labels), CLASSIFICATION_COUNT))
    for i in np.arange(len(labels)):
        onehots[i, labels[i]] = 1
    return onehots


def CoordAtt(x, reduction=32):
    def coord_act(x):
        tmpx = tf.nn.relu6(x + 3) / 6
        x = x * tmpx
        return x

    x_shape = x.get_shape().as_list()
    [b, h, w, c] = x_shape
    x_h = AvgPool2D(pool_size=(1, w), strides=1)(x)
    x_w = AvgPool2D(pool_size=(h, 1), strides=1)(x)
    x_w = tf.transpose(x_w, [0, 2, 1, 3])

    y = tf.concat([x_h, x_w], axis=1)
    mip = max(8, c // reduction)
    y = Conv2D(mip, (1, 1), strides=1, activation=coord_act, name='ca_conv1')(y)

    x_h, x_w = tf.split(y, num_or_size_splits=2, axis=1)
    x_w = tf.transpose(x_w, [0, 2, 1, 3])
    a_h = Conv2D(c, (1, 1), strides=1, activation=tf.nn.sigmoid, name='ca_conv2')(x_h)
    a_w = Conv2D(c, (1, 1), strides=1, activation=tf.nn.sigmoid, name='ca_conv3')(x_w)

    out = x * a_h * a_w

    return out


class ResnetBlock(Model):

    def __init__(self, filters, strides=1, residual_path=False):
        super(ResnetBlock, self).__init__()
        self.filters = filters
        self.strides = strides
        self.residual_path = residual_path

        self.c1 = Conv2D(filters, (3, 3), strides=strides, padding='same', use_bias=False)
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')

        self.c2 = Conv2D(filters, (3, 3), strides=1, padding='same', use_bias=False)
        self.b2 = BatchNormalization()

        # residual_path为True时，对输入进行下采样，即用1x1的卷积核做卷积操作，保证x能和F(x)维度相同，顺利相加
        if residual_path:
            self.down_c1 = Conv2D(filters, (1, 1), strides=strides, padding='same', use_bias=False)
            self.down_b1 = BatchNormalization()

        self.a2 = Activation('relu')

    def call(self, inputs):
        residual = inputs  # residual等于输入值本身，即residual=x
        # 将输入通过卷积、BN层、激活层，计算F(x)
        x = self.c1(inputs)
        x = self.b1(x)
        x = self.a1(x)

        x = self.c2(x)
        y = self.b2(x)

        if self.residual_path:
            residual = self.down_c1(inputs)
            residual = self.down_b1(residual)

        out = self.a2(CoordAtt(y + residual))  # 最后输出的是两部分的和，即F(x)+x或F(x)+Wx,再过激活函数
        return out


class ResNet18(Model):

    def __init__(self, block_list, initial_filters=64):  # block_list表示每个block有几个卷积层
        super(ResNet18, self).__init__()
        self.num_blocks = len(block_list)  # 共有几个block
        self.block_list = block_list
        self.out_filters = initial_filters
        self.c1 = Conv2D(self.out_filters, (3, 3), strides=1, padding='same', use_bias=False)
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')
        self.blocks = tf.keras.models.Sequential()
        # 构建ResNet网络结构
        for block_id in range(len(block_list)):  # 第几个resnet block
            for layer_id in range(block_list[block_id]):  # 第几个卷积层

                if block_id != 0 and layer_id == 0:  # 对除第一个block以外的每个block的输入进行下采样
                    block = ResnetBlock(self.out_filters, strides=2, residual_path=True)
                else:
                    block = ResnetBlock(self.out_filters, residual_path=False)
                self.blocks.add(block)  # 将构建好的block加入resnet
            self.out_filters *= 2  # 下一个block的卷积核数是上一个block的2倍
        self.p1 = tf.keras.layers.GlobalAveragePooling2D()
        self.f1 = tf.keras.layers.Dense(34, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2())

    def call(self, inputs):
        x = self.c1(inputs)
        x = self.b1(x)
        x = self.a1(x)
        x = self.blocks(x)
        x = self.p1(x)
        y = self.f1(x)
        return y


model = ResNet18([2, 2, 2, 2])

model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
              metrics=['categorical_accuracy'])

if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)

print("装载训练数据...")
# 获取训练集的特征矩阵、标签向量
x_train, y_train = load_data(TRAIN_DIR)
# 对训练集的特征矩阵进行正则化
x_train = normalize_data(x_train)
# 对训练集的标签向量执行独热编码
y_train = onehot_labels(y_train)
# 探查训练集
print("装载%d条数据，每条数据%d个特征" % (x_train.shape))
x_train = x_train.reshape(x_train.shape[0], IMAGE_WIDTH, IMAGE_HEIGHT, 1)  # 给数据增加一个维度，使数据和网络结构匹配

# 获取训练集的总样本数
train_samples_count = len(x_train)
train_indicies = np.arange(train_samples_count)
# 获得打乱的索引序列
np.random.shuffle(train_indicies)

print("装载测试数据...")
# 获取测试集的特征矩阵、标签向量
x_test, y_test = load_data(TEST_DIR)
# 对测试集的特征矩阵进行同样（同训练集）的正则化
x_test = normalize_data(x_test)
# 对测试集的标签向量执行独热编码
y_test = onehot_labels(y_test)
# 探查测试集
print("装载%d条数据，每条数据%d个特征" % (x_test.shape))
# 天花板取整（np.ceil），获取迭代次数（此处，就是批次）
iters = int(np.ceil(train_samples_count / 32))
x_test = x_test.reshape(x_test.shape[0], IMAGE_WIDTH, IMAGE_HEIGHT, 1)

print("Training...")
tf.config.experimental_run_functions_eagerly(True)
history = model.fit(x_train, y_train, batch_size=32, epochs=30, validation_data=(x_test, y_test), validation_freq=1,
                    callbacks=[cp_callback])

model.save_weights(save_path)

# 显示训练集和验证集的acc和loss曲线
acc = history.history['categorical_accuracy']
val_acc = history.history['val_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
