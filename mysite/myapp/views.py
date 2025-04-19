from django.shortcuts import render,redirect
from django.core.files.storage import default_storage
# Create your views here.
import pandas as pd
from django.contrib.auth import authenticate, login,logout
from django.views.generic import TemplateView, FormView
from .forms import AdditionForm
from django.urls import reverse
from django.http import HttpResponseRedirect
from django.contrib.auth.models import User
from django.contrib.auth.hashers import make_password, check_password # to hash passwords
from django.contrib import messages
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.auth import login as auth_login, logout as auth_logout
from django.views import View
# generic class based views

class IndexView(TemplateView):
    template_name = "index.html" 
    def get_context_data(self, **kwargs):
       
        context = super().get_context_data(**kwargs) 

     
        data = {'Name': ['Alice', 'Bob', 'Charlie'], 'Age': [25, 30, 35]}
        df = pd.DataFrame(data)

       
        context['df_table'] = df.to_html(index=False)

        return context

class AdditionView(View):
    template_name = 'addation.html'
    
    def get(self,request):

        form = AdditionForm()
        return render(request , self.template_name, {"form": form})

class ResultView(View):
    template_name = 'addation.html' 

    def post(self,request):

        form =  AdditionForm(request.POST)
        result = None
        if not self.request.user.is_authenticated:

            messages.error(request, "You must be logged in to perform addition.")
            return redirect('add')  # Redisplay form with message
        
        if form.is_valid():
            num1 = form.cleaned_data['number1']
            num2 = form.cleaned_data['number2']
            result = num1 + num2

        return render(request, self.template_name, {
            'form': form,
            'result': result
        })
   

class SignUpView(View):
    template_name = 'index.html'

    def get(self, request):
        return render(request, self.template_name)

    def post(self, request):
        username = request.POST['username']
        fname = request.POST['fname']
        lname = request.POST['lname']
        email = request.POST['email']
        password = request.POST['pass1']
        confirm_password = request.POST['pass2']

        if User.objects.filter(username=username).exists():
            messages.error(request, "Username already exists.")
            return redirect('index')
        
        if User.objects.filter(email=email).exists():
            messages.error(request, "Email already registered.")
            return redirect('index')
        
        if password != confirm_password:
            messages.error(request, "Passwords do not match.")
            return redirect('index')
        
        user = User.objects.create_user(
            username=username,
            email=email,
            password=password,
            first_name=fname,
            last_name=lname
        )

        messages.success(request, "Account created successfully!")
        return redirect('index')

class LoginView(View):
    template_name = 'index.html'

    def get(self, request):
        if request.user.is_authenticated:
            return redirect('index')
        return render(request, self.template_name)

    def post(self, request):
        username = request.POST['name']
        password = request.POST['loginpassword']
        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            messages.success(request, f"Welcome back, {username}!")
            return redirect('index')
        else:
            messages.error(request, "Invalid username or password.")
            return redirect('index')

class LogoutView(View):
    def get(self, request):

        file_path = request.session.get('file_path')
        if file_path and default_storage.exists(file_path):
            default_storage.delete(file_path)
        request.session.flush()
        logout(request)
        
        messages.success(request, "You have been logged out successfully.")
        return redirect('index') 

class ForgotPasswordView(View):
    template_name = 'forgotpass.html'  

    def get(self, request):
        return render(request, self.template_name)

class CreateAccountView(View):
    template_name = 'signup.html'  

    def get(self, request):
        return render(request, self.template_name)
    
class PasswordResetView(View):
    template_name = 'forgotpass.html'  

    def get(self, request):
        return render(request, self.template_name)

    def post(self, request):
        username = request.POST.get('name')
        new_password = request.POST.get('newpassword')
        confirm_password = request.POST.get('confirmpassword')

  
        try:
            user = User.objects.get(username=username)
        except User.DoesNotExist:
            messages.error(request, "Username not found.")
            return redirect('resetpage') 
        

        if check_password(new_password, user.password):
            messages.error(request, "New password cannot be the same as the old password.")
            return redirect('resetpage')
        

        if new_password != confirm_password:
            messages.error(request, "Passwords do not match.")
            return redirect('resetpage')


        user.password = make_password(new_password)
        user.save()

        messages.success(request, "Password has been reset successfully!")
        return redirect('index')  

class DeleteAccountView(View):
    template_name = 'deleteaccount.html'  

    def get(self, request):
        return render(request, self.template_name)
    
class AccountDeleteView(View):
    template_name = 'deleteaccount.html'  
    def get(self, request):
        return render(request, self.template_name)

    def post(self, request):
        username = request.POST.get('name')

        try:
            user = User.objects.get(username=username)
        except User.DoesNotExist:
            messages.error(request, "User not found.")
            return redirect('delacc')  

       
        user.delete()
        messages.success(request, f"Account for {username} has been deleted successfully.")
        return redirect('index')  

class ProfileView(LoginRequiredMixin, TemplateView):
    template_name = "profile.html"  # Specify the template

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['username'] = self.request.user.username  # Pass the username to the template
        return context
    
# Custom 404 Error View
class Custom404View(TemplateView):
    template_name = "errors/404.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['error_message'] = "The page you're looking for doesn't exist."
        return context


# Custom 500 Error View
class Custom500View(TemplateView):
    template_name = "errors/500.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['error_message'] = "An unexpected server error occurred."
        return context