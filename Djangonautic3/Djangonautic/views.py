from django.http import HttpResponse
from django.shortcuts import render,redirect
import datetime
import math
from .mainfile import get_min_max
from .mainfile import rain,windspeed
from contact.models import Contact
from django.contrib import messages
from contact.models import Contact,Reply
import requests
from bs4 import BeautifulSoup
def homepage(request):
    all_temp = get_min_max()
    r = rain()
    speed = windspeed()
    today = datetime.date.today()
    tomorrow = today+datetime.timedelta(days=1)
    _2nd_day = today + datetime.timedelta(days=2)
    _3rd_day = today + datetime.timedelta(days=3)
    _4th_day = today + datetime.timedelta(days=4)
    print("-----------------")
    print(r)
    print("--------")
    context = {
    'today':today,
    'day':today.strftime("%A"),
    'min':math.ceil(all_temp[0][0]),
    'max':math.ceil(all_temp[1][0]),
    'tomorrow':tomorrow.strftime("%A"),
    'tomorrow_max':math.ceil(all_temp[2][0]),
    'tomorrow_min':math.ceil(all_temp[3][0]),
    '2nd_max': round(all_temp[2][0]),
    '2nd_min': round(all_temp[3][0]),
    '2nd_day':_2nd_day.strftime("%A"),
    '3rd_max': round(all_temp[4][0]),
    '3rd_min': round(all_temp[5][0]),
    '3rd_day': _3rd_day.strftime("%A"),
    '4th_max': round(all_temp[6][0]),
    '4th_min': round(all_temp[7][0]),
    '4th_day': _4th_day.strftime("%A"),
    '1wind':round(r[0][0]),
    '2wind': round(r[1][0]),
        '3wind': round(r[2][0]),
        '4wind': round(r[3][0]),
        '5wind': round(r[4][0]),
        '1speed': round(speed[0][0]),
        '2speed': round(speed[1][0]),
        '3speed': round(speed[2][0]),
        '4speed': round(speed[3][0]),
        '5speed': round(speed[4][0]),
        'contact':Contact.objects.all(),
        'reply':Reply.objects.all(),
        'wether':cloudy_or_suuny()
    }
    return render(request,'homepage.html',context)

def contact(request):
    if request.method=='GET':
        return render(request,'contact.html')
    else:
        name = request.POST.get('name')
        email = request.POST.get('email')
        msg = request.POST.get('message')
        c = Contact(name=name,email=email,message=msg)
        c.save()
        messages.info(request,"thank you for your response")
        return redirect('home')

def comment_delete(request,id):
    a = Contact.objects.get(id=id)
    a.delete()
    return redirect('home')


def reply_delete(request,id):
    a = Reply.objects.get(id=id)
    a.delete()
    return redirect('home')


def cloudy_or_suuny():
    page = requests.get(
        "https://weather.com/weather/today/l/3eb968d7a06604b522f130b07342afa0c5728ddfc8a4ed54787b8676df413142")
    soup = BeautifulSoup(page.content, "html.parser")
    all = soup.find("div", {"class": "today_nowcard-phrase"}).text
    return all
