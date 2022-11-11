import re
from django.shortcuts import render
from .forms import UserRegistrationForm
from django.contrib import messages
from .models import UserRegistrationModel
import pandas as pd
import csv
from .utility.processAlgorithms import Alprocess
algo = Alprocess()


# Create your views here.
def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})


def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, 'Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHome.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})


def UserHome(request):
    return render(request, 'users/UserHome.html', {})

def Userdata(request):
    import os
    from django.conf import settings
    path = os.path.join(settings.MEDIA_ROOT, "scholar.csv")
    df = pd.read_csv(path)
    print(df)
    df = df.to_html()
    return render(request,'users/userviewdata.html',{"data":df})

def AddData(request):
    if request.method == "POST":
        percentage = request.POST.get('percentage')
        grade_point_average = request.POST.get('grade_point_average')
        marks = request.POST.get('marks')
        annual_income = request.POST.get('annual_income')
        communication_skills = request.POST.get('communication_skills')
        test_set = [percentage,grade_point_average,marks,annual_income,communication_skills]
        from django.conf import settings
        path = settings.MEDIA_ROOT + "\\" + "scholar.csv"
        data=pd.read_csv(path,delimiter=',')
        
        x = data.iloc[:,0:5]
        y = data.iloc[:,-1]

        from sklearn.model_selection import train_test_split
        from sklearn.tree import DecisionTreeClassifier
        decisiontree = DecisionTreeClassifier()
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=1)
        decisiontree.fit(x_train, y_train)
        ypred = decisiontree.predict([test_set])
        print('y pred:',ypred)
        if ypred==[1]:
            msg = 'You Are Eligible for Scholarship'
        else:
            msg = 'You are not Eligible For Scholarship'
        return render(request, "users/adddata.html", {'msg':msg})

    return render(request, "users/adddata.html", {})


def Decision_Tree(request):
    accuracy,precission,f1,recall = algo.Proces_Decision_tree()
    return render(request ,"users/decissiontree.html",{"accuracy":accuracy,"precission":precission,"f1_score":f1, 'recall':recall})

def knn(request):
    accuracy,precission,f1,recall = algo.Knn()
    return render(request, 'users/knn.html', {"accuracy":accuracy,"precission":precission,"f1_score":f1, 'recall':recall})


def NaiveBayes(request):
    accuracy,precission,f1,recall = algo.NaiveBayes()
    return render(request, 'users/NaiveBayes.html', {"accuracy":accuracy,"precission":precission,"f1_score":f1, 'recall':recall})


                 
