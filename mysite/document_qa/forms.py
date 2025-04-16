from django import forms


class DocumentUploadForm(forms.Form):
    document = forms.FileField(
        label='Upload Document',
        widget=forms.ClearableFileInput(attrs={'class': 'form-control file-upload'})
    )
    
    userquestion = forms.CharField(
        label = 'Ask Questions',
        max_length=255,
        required=True,
        widget=forms.TextInput(attrs={'class': 'form-control user-input','placeholder': "Type your Questin here"})
    )