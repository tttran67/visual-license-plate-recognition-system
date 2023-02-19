import os
import shutil
import datetime
import json
from PIL import Image
from flask_restful import Resource
from flask import redirect, request
import sys

class UploadPicture(Resource):
    """实体名称消歧联想"""

    def get(self):
        """
            上传图片，保存到服务器本地
            文件对象保存在request.files上，并且通过前端的input标签的name属性来获取
            :return: 重定向到主页
            """
        fp = request.files.get("f1")

        if fp is not None:
            now_date = datetime.datetime.now()
            uid = now_date.strftime('%Y-%m-%d-%H-%M-%S')
            # 保存文件到服务器本地
            file = "./static/img/images/%s.jpg" % uid
            file_keep = "./static/img/keep/%s.jpg" % uid
            fp.save(file)
            shutil.copy(file, file_keep)

            # file即为需要预测的图片路径
            # Todo:
            sys.path.append("../..")
            import plate_locator
            predict = plate_locator.predict_muban(file)#"津######"  # 调用接口后的预测结果
            result = {}
            result_path = './static/result.txt'
            if os.path.getsize(result_path):
                f = open('./static/result.txt', 'r')
                js = f.read()
                result = json.loads(js)
                f.close()
            file_name = "%s.jpg" % uid
            result[file_name] = predict  # 存储预测结果
            js = json.dumps(result)
            f = open('./static/result.txt', 'w')
            f.write(js)
            f.close()

            with open(file, 'rb') as f:
                if len(f.read()) < 100:
                    os.remove(file)
                    pass
                else:
                    im = Image.open(file)
                    x, y = im.size
                    y_s = int(y * 1200 / x)

                    out = im.resize((1200, y_s), Image.ANTIALIAS)

                    uid2 = now_date.strftime('%Y-%m-%d-%H-%M-%S')
                    # 保存文件到服务器本地
                    file2 = "./static/img/images/%s.jpg" % uid2
                    if len(out.mode) == 4:
                        r, g, b, a = out.split()
                        img = Image.merge("RGB", (r, g, b))
                        img.convert('RGB').save(file2, quality=10)
                    else:
                        out.save(file2)
        else:
            print('没有选择文件')
        return redirect("/")

    def post(self):
        return self.get()
