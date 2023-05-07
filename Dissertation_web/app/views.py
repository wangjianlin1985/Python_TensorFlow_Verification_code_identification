# from django.shortcuts import render
#
# # Create your views here.
#coding=utf-8
from PIL.XVThumbImagePlugin import r
from django.shortcuts import render,render_to_response
from django.http import HttpResponse
from django import forms
from io import BytesIO

from app import gen_captcha
from app.models import User
# Create your views here.
class UserForm(forms.Form):
    username = forms.CharField(label='用户名',max_length=50)
    password = forms.CharField(label='密码',widget=forms.PasswordInput())
    email = forms.EmailField(label='邮箱')
    #checkcode = forms.CharField(label='验证码',)

def getcheck_code(request):
    # code = check_code.getCheckChar()
    # img = check_code.getImg(code)
    code,img = gen_captcha.gen_captcha_text_and_image()
    f = BytesIO()
    img.save(f, 'PNG')#将图片保存在内存中
    img.save('F:/Test/123.jpg')

    # with open('F:/Test/', 'wb') as f2:
    #     f2.write(img)

    # with open('123.jpg', 'wb') as fd:
    #     for chunk in r.iter_content():
    #         fd.write(chunk)

    request.session['check_code']=code#将文本保存在session中
    return HttpResponse(f.getvalue())

def regist(request):
    if request.method == 'POST':
        userform = UserForm(request.POST)
        if userform.is_valid():
            username = userform.cleaned_data['username']
            password = userform.cleaned_data['password']
            email = userform.cleaned_data['email']

            User.objects.create(username=username,password=password,email=email)#创建用户
            #User.save()
            return HttpResponse('regist success!!!')
    else:
        userform = UserForm()
    #return render_to_response('regist.html',{'userform':userform})
    return render(request, 'regist.html', {'userform': userform})

def login(request):
    if request.method == 'POST':
        userform = UserForm(request.POST)
        if userform.is_valid():
            username = userform.cleaned_data['username']
            password = userform.cleaned_data['password']

            user = User.objects.filter(username__exact=username,password__exact=password)

            if user:

                checkcode = request.POST.get('checkcode')
                if checkcode.upper() == request.session['check_code'].upper():#将验证码都编程大写字母再作比较

                    #return render_to_response('index.html',{'userform':userform})
                    return render(request, 'index.html', {'userform':userform})
            else:
                return HttpResponse('用户名或密码错误,请重新登录')

    else:
        userform = UserForm()
    #return render_to_response('login.html',{'userform':userform})
    return render(request, 'login.html', {'userform': userform})

def index(request):
    return render(request,'index.html')
    #return render(request, 'index.html', Context({'uf': uf}))
