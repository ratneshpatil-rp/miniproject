from django.http import HttpResponse
from django.shortcuts import render
import joblib

def home(request):
    return render(request, "home.html")

def result(request):

    cls = joblib.load('finalised_model3.sav')

    lis = []

    lis.append(request.GET['sg'])
    lis.append(request.GET['al'])
    lis.append(request.GET['sc'])
    lis.append(request.GET['hemo'])
    lis.append(request.GET['pcv'])
    lis.append(request.GET['htn'])

    print(lis)

    ans = 100-(cls.predict([lis])*100)

    return render(request, "result.html", {'ans': ans, 'lis': lis})

def aboutus(request):
    return render(request, "aboutus.html")