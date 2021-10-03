from django.http import HttpResponse
from django.shortcuts import render
from matplotlib import pyplot as plt
from django.http import HttpResponse
import projet.apps as ap
def index(request) :
    var = 1

    x = [1 , 2 , 3]
    y = [1 , 2 , 3]
    plt.plot(x,y)



    context = {"var" : plt}
    return render(request,"index.html",context)

def ind(request):
    var = random.randint(5,7)
    context = {"var" : var}

    return render(request,"inter.html",context)

global x
import random


def test(request):
    var = ""
    context = {"var" : var}

    return render(request,"test.html",context)

def update(request):
    x = ap.test()
    return render(request,"test.html",x)