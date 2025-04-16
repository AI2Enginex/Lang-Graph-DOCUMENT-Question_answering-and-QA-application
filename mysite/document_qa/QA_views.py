import os
from django.shortcuts import render
from django.views import View
from .forms import DocumentUploadForm
from django.shortcuts import render, redirect
from django.contrib import messages
from django.core.files.storage import default_storage
from django.conf import settings
from .LangGraph_QuestionAnswering import QASystemGraphExecuetion
# Create your views here.


class QATemplateView(View):

    template_name = "question_answering.html"
  
    def get(self,request):
        
        form = DocumentUploadForm()
        prompt_type = request.GET.get('prompt_type', 'key word extraction') 
        return render(request , self.template_name, 
                      {"form": form,
                       "prompt": prompt_type})


class DocumentQAExecution(View):

    template_name = "question_answering.html"

    def post(self, request):
        form = DocumentUploadForm(request.POST, request.FILES)
        result = None

        if not self.request.user.is_authenticated:
            messages.error(self.request, "Login Required!!!!!")
            return redirect('login-page')
        
        if form.is_valid():
            uploaded_file = form.cleaned_data['document']
            user_query = form.cleaned_data['userquestion']
            prompt_type = request.POST.get('prompt_type','key word extraction')
            

            print(f"user query: {user_query}")
            print(f"selected prompt: {prompt_type}")

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
            

            try:

                rag_qa = QASystemGraphExecuetion(
                    data=absolute_path, processing_delimiter='\n\n',
                    total_chunk=1000,
                    overlapping=300,embedding_model='models/embedding-001')
                result = rag_qa.answer(question=user_query,prompt=prompt_type)

            except Exception as e:
                messages.error(request, f"An error occurred: {str(e)}")

            finally:
                default_storage.delete(file_path)

            return render(request, self.template_name, {
                'form': DocumentUploadForm(),
                'answer': result,
                'prompt_type': prompt_type,
            })
        
        messages.error(request, 'Invalid form submission.')
        return render(request, self.template_name, {'form': form})

