from django.shortcuts import render
import os
from django.core.files.storage import FileSystemStorage
from project.handler import Handler

my_handler = Handler()

# Create your views here.
def homepage(request) :
    return render(request,'homepage.html')

def generate_home(request) :
    mode = 'gen'
    my_handler.set_mode(mode)
    print("Mode set to gen")
    return render(request, 'generate_home.html')

def generate_result(request) :
    input_text = request.POST.get('text_input')
    my_handler.set_text(input_text)
    success = my_handler.execute()
    if success :
        print("Image generated")
    else :
        print("Something went wrong")
    image_url = "/media/generated.jpg"
    return render(request, 'generate_result.html', {"image_url" : image_url})

def manipulate_home(request) :
    mode = 'man'
    my_handler.set_mode(mode)
    print("Mode set to man")
    return render(request, 'manipulate_home.html')

def manipulate_last(request) :
    image_input_mode = 'last'
    input_text = request.POST.get('text_input')
    my_handler.set_image_input_mode(image_input_mode)
    my_handler.set_text(input_text)
    image_url = "/media/generated.jpg"
    return render(request, 'manipulate_last.html',{"image_url" : image_url})

def manipulate_upload(request) :
    image_input_mode = 'upload'
    input_text = request.POST.get('text_input')
    my_handler.set_image_input_mode(image_input_mode)
    my_handler.set_text(input_text)
    return render(request, 'manipulate_upload.html')
    
def manipulate_result(request) :
    # If image was uploaded, the url corresponding to this view
    # would be called by POST method along with image 
    # so we need to save the image first before manipulation
    if request.method == "POST" :
        uploaded_image = request.FILES['uploaded_image']
        project_dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        file_path = "project/test/input.jpg"
        upload_file_name = os.path.join(project_dir_path, file_path)
        if os.path.exists(upload_file_name) :
            os.remove(upload_file_name)
        fs = FileSystemStorage()
        name = fs.save('input.jpg', uploaded_image)
        input_url = "/media/input.jpg"
    else :
        input_url = "/media/generated.jpg"
    output_url = "/media/manipulated.jpg"
    success = my_handler.execute()
    if success :
        print("Image generated")
    else :
        print("Something went wrong")
    return render(request, 'manipulate_result.html',{"input_url" : input_url, "output_url" : output_url})