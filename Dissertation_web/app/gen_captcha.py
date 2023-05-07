from captcha.image import ImageCaptcha  # pip3 install captcha，用于将验证码文本转化成验证码图片
import numpy as np
import matplotlib.pyplot as plt # sudo pip3 install matplotlib && sudo apt-get install python3-tk 此处为转载时添加
from PIL import Image#使用Image中的open函数打开验证码图片
import random#引用时相当于np.random

# 验证码中的字符, 就不用汉字了
number = ['0','1','2','3','4','5','6','7','8','9']
alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
ALPHABET = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
# 验证码一般都无视大小写；验证码长度4个字符
def random_captcha_text(char_set=number+alphabet+ALPHABET, captcha_size=4):
    captcha_text = []
    for i in range(captcha_size):
        c = random.choice(char_set)
        #c = np.random.choice(char_set)
        captcha_text.append(c)
    return captcha_text

# 生成字符对应的验证码
def gen_captcha_text_and_image():
    image = ImageCaptcha()#生成一个图片验证码对象

    captcha_text = random_captcha_text()
    captcha_text = ''.join(captcha_text)#将字符连接为字符串，中间间隔引号里的内容
    print(captcha_text)

    captcha = image.generate(captcha_text)#用于将验证码文本转化成验证码图片
    #image.write(captcha_text, captcha_text + '.jpg')  # 写到文件

    captcha_image = Image.open(captcha)#用python自带的Image库函数打开验证码图片
    captcha_image.save('F:/gen_captcha.jpg')
    #captcha_image = np.array(captcha_image)#将打开的验证码图片存入数组中
    return captcha_text, captcha_image

if __name__ == '__main__':
    # 测试
    text, image = gen_captcha_text_and_image()
    # print(image)
    # print(text)

    f = plt.figure()#生成一个画布
    ax = f.add_subplot(111)#其中111表示将画布分为1行1列共1块，图片在第1块
    ax.text(0.1, 0.9, text, ha='center', va='center', transform=ax.transAxes)
    plt.imshow(image)

    plt.show()