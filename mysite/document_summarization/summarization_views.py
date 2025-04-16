import os
from django.shortcuts import render
from django.views import View
from django.views.generic.edit import FormView
from django.contrib import messages
from django.urls import reverse_lazy
from .forms import DocumentUploadForm
from django.shortcuts import render, redirect
from django.contrib import messages
from django.core.files.storage import default_storage
from django.conf import settings

from .LangGraph_summarization import StuffGraphExecuetion, MapReduceGraphExecuetion
# Create your views here.

class SummarizationTemplate(View):
    template_name = 'summarization.html'

    def get(self, request):
        form = DocumentUploadForm()
        prompt_type = request.GET.get('prompt_type', 'simple')  # <-- FIXED # field for the form
        return render(request, self.template_name, {
        'form': form,
        'prompt_type': prompt_type})

    
class DocumentSummarizationView(View):
    template_name = 'summarization.html'

    # def get(self, request):
    #     form = DocumentUploadForm()
    #     return render(request, self.template_name, {'form': form})

    def post(self, request):
        form = DocumentUploadForm(request.POST, request.FILES)
        summary = None
        

        if not self.request.user.is_authenticated:
            messages.error(self.request, "Login Required!!!!!")
            return redirect('login-page') 
        
        if form.is_valid():
            uploaded_file = form.cleaned_data['document']
            prompt_type = request.POST.get('prompt_type', 'simple')

            # Check file extension
            if not uploaded_file.name.lower().endswith('.pdf'):
                messages.error(request, 'Only PDF files are supported.')
                return render(request, self.template_name, {
                    'form': form,
                    'summary': None,
                    'prompt_type': prompt_type,
                })

            # Save and process the valid PDF
            file_path = default_storage.save(uploaded_file.name, uploaded_file)
            absolute_path = os.path.join(settings.MEDIA_ROOT, file_path)

            print('selected prompt type: ', prompt_type)
            query = "Summarize this document briefly."

            try:
                if prompt_type == 'simple':
                    rag = StuffGraphExecuetion(
                        data=absolute_path,
                        processing_delimiter='\n\n',
                        total_chunk=1000,
                        overlapping=300,
                        embedding_model='models/embedding-001'
                    )
                else:
                    rag = MapReduceGraphExecuetion(
                        data=absolute_path,
                        processing_delimiter='\n\n',
                        total_chunk=1000,
                        overlapping=300,
                        embedding_model='models/embedding-001'
                    )

                summary = rag.summarize(query=query)
                if summary:
                    messages.success(request, 'Document summarized successfully.')

            except Exception as e:
                messages.error(request, f"An error occurred: {str(e)}")

            finally:
                default_storage.delete(file_path)

            return render(request, self.template_name, {
                'form': DocumentUploadForm(),
                'summary': summary,
                'prompt_type': prompt_type,
            })

        messages.error(request, 'Invalid form submission.')
        return render(request, self.template_name, {'form': form})