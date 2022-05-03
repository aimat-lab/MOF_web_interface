from django.shortcuts import render
from AI import do_all
import os
#import pdb


def index(request):
    context = {}
    return render(request, 'test_ai/index.html', context)


def upload(request):
    result = {}
    if request.method == "POST":
        upload_file = request.FILES['document']

        startpath = os.getcwd()

        if startpath.endswith("MOF_web_interface"):
            startpath+="/AI"
        elif startpath.endswith("AI"):
            pass

        file_content = upload_file.read()
        cif_filename = upload_file.name

        outfile=open("%s/temp_cif_files/%s"%(startpath, cif_filename), "bw")
        outfile.write(file_content)
        outfile.close()
        print("writing new cif file to %s/temp_cif_files/%s"%(startpath, cif_filename))

        #pdb.set_trace()
        try:
            predictions = do_all.do_all("%s/temp_cif_files/%s"%(startpath, cif_filename), startpath)
            result = {'file_content': file_content, 'predictions': predictions,'temperature': predictions[0][0], 'time': predictions[1][0], 'solvent': predictions[2][0], 'additive': predictions[3][0], 'temperature_cert': predictions[0][1], 'time_cert': predictions[1][1], 'solvent_cert': predictions[2][1], 'additive_cert': predictions[3][1], 'png_string': predictions[2][2]}
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except Exception as e:
            print(str(e))
            print("The AI threw an error!")
            result = {'file_content': "ERROR", 'predictions': "ERROR", 'temperature': "ERROR",
                    'time': "ERROR", 'solvent': "ERROR", 'additive': "ERROR"}

    return render(request, 'test_ai/upload.html', result)





