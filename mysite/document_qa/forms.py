from django import forms

class DocumentOnlyForm(forms.Form):
    document = forms.FileField(
        label='',
        widget=forms.ClearableFileInput(attrs={'class': 'form-control file-upload'})
    )


class DocumentUploadForm(DocumentOnlyForm):
    userquestion = forms.CharField(
        label='Ask Questions',
        max_length=255,
        required=True,
        widget=forms.TextInput(attrs={'class': 'form-control user-input','placeholder': "Type your question here"})
    )
