import os
import sys
import time
from datetime import datetime
import uuid
import pandas as pd

print(__file__)
# own libraries
from AI import MOF_RAC_example
from AI import model_additive
from AI import model_solvent
from AI import model_temperature
from AI import model_time
from AI import final_prediction


def do_all(cif_filename):


    # get a random name for the MOF
    now = datetime.now()
    date_string = now.strftime("%Y_%m_%d_%H_%M_%S")
    random_string = uuid.uuid4()
    MOF_random_name = "%s_%s"%(date_string, random_string)

    # log everything
    logfile = open("mof_predictions.log", "a")
    logfile.write("%s %s\n"%(MOF_random_name, cif_filename))
    logfile.close()


    ###Download the uploaded cif file by the user and keep it in the following directory named cif###
    if not os.path.exists("cif_files/"):
        os.makedirs("cif_files/")

    if os.path.exists("%s"%(cif_filename)):
        if not os.path.exists("cif_files/%s"%(MOF_random_name)):
            os.makedirs("cif_files/%s"%(MOF_random_name))
        os.system("cp %s cif_files/%s/mof.cif"%(cif_filename, MOF_random_name)) #######Copy the downloaded cif file to the cif directory
    else:
        print("ERROR: file not found: %s"%(cif_filename))
        return(None, None, None, None)


    ################Calculation of RAC Features of the downloaded cif file using Molsimplify######################

    #df_new = MOF_RAC_example.compute_features(MOF_random_name)
    df_new = pd.read_csv('additional_data/full_featurization_frame.csv')

    #####################################################################

    ##################Load the ML Models and run  (Now only RF model is loaded)##################
    predictions_temperature = model_temperature.model_temperature(MOF_random_name, df_new)
    predictions_time = model_time.model_time(MOF_random_name, df_new)
    predictions_additive = model_additive.model_additive(MOF_random_name, df_new)
    predictions_solvent = model_solvent.model_solvent(MOF_random_name, df_new)

    #############################################################################

    ################################Final_Prediction############################

    predictions = final_prediction.final_prediction(MOF_random_name, predictions_temperature, predictions_time, predictions_solvent, predictions_additive)   #This will print the final Prediction
    temperature, time, solvent, additive_string = predictions

    ############################################################

    return(temperature, time, solvent, additive_string)



if __name__ == "__main__":

    cif_filename = sys.argv[1]
    temperature, time, solvent, additive_string = do_all(cif_filename)





