from PIL import Image,ImageDraw,ImageFont,ImageFilter
import random
import string

ttf='C:\Windows\Fonts'

def getCheckChar():
    ran=string.ascii_letters+string.digits  #ascii_letters 生成随机字母（大小写），digits生成数字
    check_char=''
    for i in range(4):
        check_char+=random.choice(ran)
    print(check_char)
    return check_char

def getImg(code):
    img=Image.new('RGB',(120,30),(255,255,255))
    draw=ImageDraw.Draw(img)
    font = ImageFont.truetype('C:\Windows\Fonts\Monaco.ttf', 18)
    code=code
    color=random.randint(50, 150), random.randint(50, 150), random.randint(50, 150)
    for t in range(4):
        draw.text((28*t,0),code[t],color,font)
    # draw.text((60,0),code,color,font)
    #干扰点
    chance=min(100, max(0, int(2)))
    for w in range(120):
        for h in range(30):
            tmp = random.randint(0, 100)
            if tmp > 100 - chance:
                draw.point((w, h), fill=(0, 0, 0))
    #干扰线
    for i in range(3):
        # 起始点
        begin = (random.randint(0, 120), random.randint(0, 30))
        # 结束点
        end = (random.randint(0, 120), random.randint(0, 30))
        draw.line([begin, end], fill=(0, 0, 0))
    #图形扭曲参数
    params = [1 - float(random.randint(1, 2)) / 100,
              0,
              0,
              0,
              1 - float(random.randint(1, 10)) / 100,
              float(random.randint(1, 2)) / 500,
              0.001,
              float(random.randint(1, 2)) / 500
              ]
    img = img.transform((120,30), Image.PERSPECTIVE,params, Image.BILINEAR)
    img=img.filter(ImageFilter.EDGE_ENHANCE_MORE)
    # img.save(''.join(code)+'.jpg','jpeg')
    return img

if __name__ == '__main__':
    code=getCheckChar()
    getImg(code)