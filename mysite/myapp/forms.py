from django import forms

class AdditionForm(forms.Form):
    number1 = forms.IntegerField(label='Number 1', required=True)
    number2 = forms.IntegerField(label='Number 2', required=True)
