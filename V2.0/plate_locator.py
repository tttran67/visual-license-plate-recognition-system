import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import util
import os
import tensorflow as tf
import predict_cn
import predict_en

def split(word_path):
    HSV_MIN_BLUE_H = 100  # HSV中蓝色分量最小范围值
    HSV_MAX_BLUE_H = 140  # HSV中蓝色分量最大范围值
    MAX_SV = 255
    MIN_SV = 95
    # plate_file_path = "images/plate1.jpg"
    plate_file_path = word_path
    plate_image = cv.imread(plate_file_path)

    hsv_image = cv.cvtColor(plate_image, cv.COLOR_BGR2HSV)
    h_split, s_split, v_split = cv.split(hsv_image)  # 将H,S,V分量分别放到三个数组中
    rows, cols = h_split.shape

    binary_image = np.zeros((rows, cols), dtype=np.uint8)

    # 将满足蓝色背景的区域，对应索引的颜色值置为255，其余置为0，从而实现二值化
    for row in np.arange(rows):
        for col in np.arange(cols):
            H = h_split[row, col]
            S = s_split[row, col]
            V = v_split[row, col]
            # 在蓝色值域区间，且满足S和V的一定条件
            if (H >= HSV_MIN_BLUE_H and H <= HSV_MAX_BLUE_H) \
                    and (S >= MIN_SV and S <= MAX_SV) \
                    and (V >= MIN_SV and V <= MAX_SV):
                binary_image[row, col] = 255

    # 执行闭操作，使相邻区域连成一片
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (10, 3))
    morphology_image = cv.morphologyEx(binary_image, cv.MORPH_CLOSE, kernel)

    contours, _ = cv.findContours(morphology_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    verified_plates = []
    for i in np.arange(len(contours)):
        if util.verify_plate_sizes(contours[i]):
            output_image = util.rotate_plate_image(contours[i], plate_image)
            output_image = util.unify_plate_image(output_image)
            verified_plates.append(output_image)
   # for i in np.arange(len(verified_plates)):
    #    cv.imshow("", verified_plates[i])
        #cv.waitKey()

    #cv.destroyAllWindows()

    # 读取待检测图片
    origin_image = verified_plates[0]
    # 复制一张图片，在复制图上进行图像操作，保留原图
    image = origin_image.copy()

    # # 读取标准图片
    # biaozhun_image = cv.imread('./images/car.jpg')
    #
    # #获得待检测图片的尺寸
    # biaozhun_height, biaozhun_width,_ = biaozhun_image.shape
    # print("---------------------------------------")
    # print(biaozhun_height,biaozhun_width)
    # # 将模板resize至与图像一样大小
    image = cv.resize(image, (294, 83))


    # plt显示彩色图片
    def plt_show0(img):
        # cv2与plt的图像通道不同：cv2为[b,g,r];plt为[r, g, b]
        b, g, r = cv.split(img)
        img = cv.merge([r, g, b])
        plt.imshow(img)
        plt.show()


    # plt显示灰度图片
    def plt_show(img):
        plt.imshow(img, cmap='gray')
        plt.show()


    # 图像去噪灰度处理
    def gray_guss(image):
        # 高斯去噪
        image = cv.GaussianBlur(image, (3, 3), 0)
        # 灰度图
        gray_image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        return gray_image


    # 1、读取+预处理图像（高斯去噪、灰度处理、二值化）
    # 图像去噪灰度处理
    gray_image = gray_guss(image)
    # 图像阈值化操作——获得二值化图
    ret, image = cv.threshold(gray_image, 0, 255, cv.THRESH_OTSU)
    plt_show(image)

    # 2、分割字符
    white = []  # 记录每一列的白色像素总和
    black = []  # ..........黑色.......
    height = image.shape[0]
    width = image.shape[1]
    white_max = 0
    black_max = 0
    # 计算每一列的黑白色像素总和
    for i in range(width):
        s = 0  # 这一列白色总数
        t = 0  # 这一列黑色总数
        for j in range(height):
            if image[j][i] == 255:
                s += 1
            if image[j][i] == 0:
                t += 1
        white_max = max(white_max, s)
        black_max = max(black_max, t)
        white.append(s)
        black.append(t)
        # print(s)
        # print(t)

    arg = False  # False表示白底黑字；True表示黑底白字
    if black_max > white_max:
        arg = True


    # 分割图像
    def find_end(start_):
        end_ = start_ + 1
        for m in range(start_ + 1, width - 1):
            if (black[m] if arg else white[m]) > (
            0.95 * black_max if arg else 0.95 * white_max):  # 0.95这个参数请多调整，对应下面的0.05（针对像素分布调节）
                end_ = m
                break
        return end_


    n = 1
    start = 1
    end = 2
    word = []
    while n < width - 2:
        n += 1
        if (white[n] if arg else black[n]) > (0.05 * white_max if arg else 0.05 * black_max):
            # 上面这些判断用来辨别是白底黑字还是黑底白字
            # 0.05这个参数请多调整，对应上面的0.95
            start = n
            end = find_end(start)
            n = end
            if end - start > 5:
                cj = image[1:height, start:end]
                cj = cv.resize(cj, (15, 30))
                word.append(cj)

    print(len(word))


    def chuli(img):
        img0 = img.copy()
        # # 膨胀操作，使字符膨胀为一个近似的整体，为查找轮廓做准备
        # kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
        # img = cv.dilate(img, kernel)
        # #plt_show(img)

        # 查找轮廓
        contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        words = []
        word_images = []
        # 对所有轮廓逐一操作
        for item in contours:
            word = []
            rect = cv.boundingRect(item)
            x = rect[0]
            y = rect[1]
            weight = rect[2]
            height = rect[3]
            word.append(x)
            word.append(y)
            word.append(weight)
            word.append(height)
            words.append(word)
        # 排序，车牌号有顺序。words是一个嵌套列表
        words = sorted(words, key=lambda s: s[0], reverse=False)
        print(words)

        # word中存放轮廓的起始点和宽高

        for word in words:
            # 筛选字符的轮廓
            if (word[3] > (word[2] * 1.2)) and (word[3] < (word[2] * 3.5)):
                # if (word[3] > (word[2] * 1.0)) and (word[3] < (word[2] * 3.5)) :
                splite_image = img0[word[1]:word[1] + word[3], word[0]:word[0] + word[2]]
                return splite_image
        return img0


    def chuli_for_hanzi(img):
        img0 = img.copy()
        # 膨胀操作，使字符膨胀为一个近似的整体，为查找轮廓做准备
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        img = cv.dilate(img, kernel)
        plt_show(img)

        # 查找轮廓
        contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        words = []
        word_images = []
        # 对所有轮廓逐一操作
        for item in contours:
            word = []
            rect = cv.boundingRect(item)
            x = rect[0]
            y = rect[1]
            weight = rect[2]
            height = rect[3]
            word.append(x)
            word.append(y)
            word.append(weight)
            word.append(height)
            words.append(word)
        # 排序，车牌号有顺序。words是一个嵌套列表
        words = sorted(words, key=lambda s: s[0], reverse=False)
        print(words)

        # word中存放轮廓的起始点和宽高
        for word in words:
            # 筛选字符的轮廓
            if (word[3] > (word[2] * 1.2)) and (word[3] < (word[2] * 3.5)):
                # if (word[3] > (word[2] * 1.0)) and (word[3] < (word[2] * 3.5)) :
                splite_image = img0[word[1]:word[1] + word[3], word[0]:word[0] + word[2]]
                return splite_image
        return img0


    def chuli_for_point(img):
        img0 = img.copy()
        # 膨胀操作，使字符膨胀为一个近似的整体，为查找轮廓做准备
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        img = cv.dilate(img, kernel)
        # plt_show(img)

        # 查找轮廓
        contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        words = []
        word_images = []
        # 对所有轮廓逐一操作
        for item in contours:
            word = []
            rect = cv.boundingRect(item)
            x = rect[0]
            y = rect[1]
            weight = rect[2]
            height = rect[3]
            word.append(x)
            word.append(y)
            word.append(weight)
            word.append(height)
            words.append(word)
        # 排序，车牌号有顺序。words是一个嵌套列表
        words = sorted(words, key=lambda s: s[0], reverse=False)
        print(words)

        # word中存放轮廓的起始点和宽高
        for word in words:
            # 筛选字符的轮廓
            if (word[3] > (word[2] * 1.0)) and (word[3] < (word[2] * 3.5)):
                return False
        return True


    # 去除中间分割点
    temp_img = word[2]
    if (chuli_for_point(temp_img)):
        word.pop(2)

    for i in range(len(word)):
        temp_img = word[i]
        if i == 0:
            word[i] = chuli_for_hanzi(temp_img)
        else:
            word[i] = chuli(temp_img)

    for i, j in enumerate(word):
        plt.subplot(1, 8, i + 1)
        plt.imshow(word[i], cmap='gray')
    plt.show()
    return word






# 模版匹配----------------------------------------------------------------------------

def predict_muban(word_path):
    word=split(word_path)
    # 读取一个文件夹下的所有图片，输入参数是文件名，返回模板文件地址列表
    def read_directory(directory_name):
        referImg_list = []
        for filename in os.listdir(directory_name):
            referImg_list.append(directory_name + "/" + filename)
        return referImg_list

    # 获得中文模板列表（只匹配车牌的第一个字符）
    def get_chinese_words_list():
        chinese_words_list = []
        for i in range(34, 64):
            # 将模板存放在字典中
            c_word = read_directory('./refer1/' + template[i])
            chinese_words_list.append(c_word)
        return chinese_words_list

    # 获得英文模板列表（只匹配车牌的第二个字符）
    def get_eng_words_list():
        eng_words_list = []
        for i in range(10, 34):
            e_word = read_directory('./refer1/' + template[i])
            eng_words_list.append(e_word)
        return eng_words_list

    # 获得英文和数字模板列表（匹配车牌后面的字符）
    def get_eng_num_words_list():
        eng_num_words_list = []
        for i in range(0, 34):
            word = read_directory('./refer1/' + template[i])
            eng_num_words_list.append(word)
        return eng_num_words_list

    # 读取一个模板地址与图片进行匹配，返回得分
    def template_score(template, image):
        # 将模板进行格式转换
        template_img = cv.imdecode(np.fromfile(template, dtype=np.uint8), 1)
        template_img = cv.cvtColor(template_img, cv.COLOR_RGB2GRAY)
        # 模板图像阈值化处理——获得黑白图
        ret, template_img = cv.threshold(template_img, 0, 255, cv.THRESH_OTSU)
        #     height, width = template_img.shape
        #     image_ = image.copy()
        #     image_ = cv2.resize(image_, (width, height))
        image_ = image.copy()
        # 获得待检测图片的尺寸
        height, width = image_.shape
        # 将模板resize至与图像一样大小
        template_img = cv.resize(template_img, (width, height))
        # 模板匹配，返回匹配得分
        result = cv.matchTemplate(image_, template_img, cv.TM_CCOEFF)
        return result[0][0]

    # 对分割得到的字符逐一匹配
    def template_matching(word_images):
        results = []
        for index, word_image in enumerate(word_images):
            if index == 0:
                best_score = []
                for chinese_words in chinese_words_list:
                    score = []
                    for chinese_word in chinese_words:
                        result = template_score(chinese_word, word_image)
                        score.append(result)
                    best_score.append(max(score))
                i = best_score.index(max(best_score))
                # print(template[34+i])
                r = template[34 + i]
                results.append(r)
                continue
            if index == 1:
                best_score = []
                for eng_word_list in eng_words_list:
                    score = []
                    for eng_word in eng_word_list:
                        result = template_score(eng_word, word_image)
                        score.append(result)
                    best_score.append(max(score))
                i = best_score.index(max(best_score))
                # print(template[10+i])
                r = template[10 + i]
                results.append(r)
                continue
            else:
                best_score = []
                for eng_num_word_list in eng_num_words_list:
                    score = []
                    for eng_num_word in eng_num_word_list:
                        result = template_score(eng_num_word, word_image)
                        score.append(result)
                    best_score.append(max(score))
                i = best_score.index(max(best_score))
                # print(template[i])
                r = template[i]
                results.append(r)
                continue
        return results

    # 准备模板(template[0-9]为数字模板；)
    template = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
                'X', 'Y', 'Z',
                '藏', '川', '鄂', '甘', '赣', '贵', '桂', '黑', '沪', '吉', '冀', '津', '晋', '京', '辽', '鲁', '蒙', '闽', '宁',
                '青', '琼', '陕', '苏', '皖', '湘', '新', '渝', '豫', '粤', '云', '浙']
    chinese_words_list = get_chinese_words_list()
    eng_words_list = get_eng_words_list()
    eng_num_words_list = get_eng_num_words_list()

    word_images_ = word.copy()
    # 调用函数获得结果
    result = template_matching(word_images_)
    print(result)
    # "".join(result)函数将列表转换为拼接好的字符串，方便结果显示
    print("".join(result))
    return "".join(result)






# Resnet预测

def resnet_predict_cn(image_i,model):
    IMAGE_WIDTH = 24
    IMAGE_HEIGHT = 48
    CLASSIFICATION_COUNT = 31
    CHINESE_LABELS = [
        'chuan', 'e', 'gan', 'gan1', 'gui', 'gui1', 'hei', 'hu', 'ji', 'jin',
        'jing', 'jl', 'liao', 'lu', 'meng', 'min', 'ning', 'qing', 'qiong', 'shan',
        'su', 'sx', 'wan', 'xiang', 'xin', 'yu', 'yu1', 'yue', 'yun', 'zang',
        'zhe']


    print('=====================================')
    digit_image = predict_cn.load_image0(image_i, IMAGE_WIDTH, IMAGE_HEIGHT)
    digit_image = digit_image.astype('float')
    digit_image = digit_image.reshape(IMAGE_WIDTH, IMAGE_HEIGHT, 1)

    x_predict = digit_image[tf.newaxis, ...]
    print(x_predict.shape)

    result = model(x_predict, training=False)
    pred = tf.argmax(result, axis=1)
    print('\n')
    print(CHINESE_LABELS[int(pred)])
    return CHINESE_LABELS[int(pred)]



def resnet_predict_en(image_i,model):
    ### 模型定义开始
    IMAGE_WIDTH = 20
    IMAGE_HEIGHT = 20
    CLASSIFICATION_COUNT = 34
    ENGLISH_LABELS = [
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
        'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
        'W', 'X', 'Y', 'Z']

    print('=====================================')
    digit_image = predict_en.load_image0(image_i, IMAGE_WIDTH, IMAGE_HEIGHT)
    digit_image = digit_image.astype('float')
    digit_image = digit_image.reshape(IMAGE_WIDTH, IMAGE_HEIGHT, 1)

    x_predict = digit_image[tf.newaxis, ...]
    print(x_predict.shape)

    result = model(x_predict, training=False)
    pred = tf.argmax(result, axis=1)
    print('\n')
    print(int(pred))
    return ENGLISH_LABELS[int(pred)]




def resnet_predict(word_path):
    word_images = split(word_path)
    print(len(word_images))
    results = []
    #加载en模型
    en_model = predict_en.ResNet18([2, 2, 2, 2])
    ENGLISH_MODEL_PATH = "checkpoint_en/ResNet18.ckpt"
    en_model.load_weights(ENGLISH_MODEL_PATH)
    #加载cn模型
    cn_model = predict_cn.ResNet18([2, 2, 2, 2])
    Chinese_MODEL_PATH = './weights_cn/'
    cn_model.load_weights(Chinese_MODEL_PATH)

    for index, word_image in enumerate(word_images):
        if index == 0:
            r=resnet_predict_cn(word_images[index],cn_model)
            results.append(r)
        else:

            r = resnet_predict_en(word_images[index],en_model)
            results.append(r)
    print(results)
    # "".join(result)函数将列表转换为拼接好的字符串，方便结果显示
    print("".join(results))
    return "".join(results)


if __name__=="__main__":
    #print("mb:"+predict_muban("images/p4.jpg"))

    print("dl:"+resnet_predict("images/p4.jpg"))





