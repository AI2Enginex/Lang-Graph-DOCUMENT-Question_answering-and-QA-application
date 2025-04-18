from django import forms

class DocumentUploadForm(forms.Form):
    document = forms.FileField(label='',widget=forms.ClearableFileInput(attrs={'class': 'form-control'}))
