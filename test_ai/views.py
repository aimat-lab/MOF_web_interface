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
        print(startpath)
        if startpath.endswith("MOF_web_interface"):
            startpath+="/AI"
        elif startpath.endswith("AI"):
            pass

        #os.chdir("AI")
        file_content = upload_file.read()
        #cif_filename = "tests/mof1.cif" # for testing
        cif_filename = upload_file.name
        outfile=open("%s/temp_cif_files/%s"%(startpath, cif_filename), "bw")
        outfile.write(file_content)
        outfile.close()
        result=None

        try:
            predictions = do_all.do_all("%s/temp_cif_files/%s"%(startpath, cif_filename), startpath)

            result = {'file_content': file_content, 'predictions': predictions, 'temperature': predictions[0], 'time': predictions[1], 'solvent': predictions[2], 'additive': predictions[3]}
        except Exception as e:
            print(e.message)
            print("The AI threw an error!")
            result = {'file_content': "ERROR", 'predictions': "ERROR", 'temperature': "ERROR",
             'time': "ERROR", 'solvent': "ERROR", 'additive': "ERROR"}

        #os.chdir(startpath)
    return render(request, 'test_ai/upload.html', result)





