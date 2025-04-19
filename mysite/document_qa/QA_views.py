import os
import re
from django.shortcuts import render, redirect
from django.views import View
from django.contrib import messages
from django.core.files.storage import default_storage
from django.conf import settings
from django.http import JsonResponse
from .forms import DocumentOnlyForm
from .forms import DocumentUploadForm
from .LangGraph_QuestionAnswering import QASystemGraphExecuetion



# function for cleaning the
# LLM'S response 
def clean_answer(text):
    """
    This function takes in a string input and removes any special characters
    """
    # Keep letters, numbers, whitespace, and standard punctuation
    return re.sub(r"[^\w\s.,:;!?'-]", '', text)



# class for rendering the
# template with the initial form
class QATemplateView(View):


    """
    The class helps in rendering the template with the form

    """
    template_name = "question_answering.html"

    def get(self, request):
        form = DocumentUploadForm()
        prompt_type = request.GET.get('prompt_type', 'key word extraction')
        return render(request, self.template_name, {
            "form": form,
            "prompt": prompt_type
        })



# class for uploading the
# document and storing the document 
# in the session later to beused
# for questin and answering

class UploadDocumentView(View):

    """
    The class allows the user to upload a "pdf" document
    and stores the document in a session. later the document
    can be used for multiple question and answering task, the 
    document remains selected in the session until the user
    is logged in this helps for multiple QA task without any need of reuploading the document

    """
    def post(self, request):
        form = DocumentOnlyForm(request.POST, request.FILES)

        if not request.user.is_authenticated:
            return JsonResponse({'error': 'Authentication required'}, status=401)

        if form.is_valid() and 'document' in request.FILES:
            uploaded_file = form.cleaned_data['document']

            if not uploaded_file.name.lower().endswith('.pdf'):
                return JsonResponse({'error': 'Only PDF files are supported.'}, status=400)

            file_path = default_storage.save(uploaded_file.name, uploaded_file)
            request.session['file_path'] = file_path
            return JsonResponse({'message': 'File uploaded', 'file_name': uploaded_file.name})

        return JsonResponse({'error': 'Invalid upload'}, status=400)


# class for generating the LLM'S
# response and renders the response
# at the frontend
class DocumentQAExecution(View):


    """
    The class handles the entire flow of Document QA
    it keeps the track of the file form the session 
    which helps with the multiple QA with the same document.

    """
    template_name = "question_answering.html"

    def post(self, request):
        form = DocumentUploadForm(request.POST, request.FILES)
        result = None

        if not request.user.is_authenticated:
            messages.error(request, "Login Required!!!!!")
            return redirect('login-page')

        prompt_type = request.POST.get('prompt_type', 'key word extraction')
        file_path = request.session.get('file_path')

        if not file_path:
            messages.error(request, 'No document uploaded.')
            return render(request, self.template_name, {
                'form': form,
                'prompt_type': prompt_type,
            })

        user_query = request.POST.get('userquestion')

        if not user_query:
            messages.error(request, 'Please enter a question.')
            return render(request, self.template_name, {
                'form': form,
                'prompt_type': prompt_type,
            })

        absolute_path = os.path.join(settings.MEDIA_ROOT, file_path)

        try:
            rag_qa = QASystemGraphExecuetion(
                data=absolute_path,
                processing_delimiter='\n\n',
                total_chunk=1000,
                overlapping=300,
                embedding_model='models/embedding-001'
            )
            result = rag_qa.answer(question=user_query, prompt=prompt_type)
            cleaned_result = clean_answer(result)

        except Exception as e:
            messages.error(request, f"An error occurred: {str(e)}")

        return render(request, self.template_name, {
            'form': DocumentUploadForm(),
            'answer': cleaned_result,
            'prompt_type': prompt_type,
        })



