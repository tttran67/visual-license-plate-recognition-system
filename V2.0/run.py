from flask import Flask
from api import bp
import json

app = Flask(__name__)
app.register_blueprint(bp, url_prefix='/')


if __name__ == '__main__':
    '''
    result = {"2021-01-28-16-01-00.jpg": "津A11345"}
    js = json.dumps(result)
    file = open('./static/result.txt', 'w')
    file.write(js)
    file.close()'''
    # 生产状态可以把debug调成False
    app.run(host='0.0.0.0', port=8000, debug=True)
