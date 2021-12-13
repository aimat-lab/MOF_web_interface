from django.shortcuts import render
from AI import do_all
import os



def index(request):
    context = {}
    return render(request, 'test_ai/index.html', context)


def upload(request):
    result = {}
    if request.method == "POST":
        upload_file = request.FILES['document']
        print(upload_file.name)
        print(upload_file.size)

        startpath = os.getcwd()
        os.chdir("AI")
        file_content = upload_file.read()
        #cif_filename = "tests/mof1.cif" # for testing
        cif_filename = upload_file.name
        outfile=open("temp_cif_files/%s"%(cif_filename), "bw")
        outfile.write(file_content)
        outfile.close()
        predictions = do_all.do_all("temp_cif_files/%s"%(cif_filename))
        print(predictions)
        result = {'file_content': file_content, 'predictions': predictions}
        os.chdir(startpath)
    return render(request, 'test_ai/upload.html', result)





