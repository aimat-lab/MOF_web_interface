import os
import sys
import time
from datetime import datetime
import uuid
import pandas as pd
import shutil

print(__file__)
# own libraries
try:
    from AI import MOF_RAC_example
    from AI import MOF_prediction_models
    from AI import final_prediction
except:
    import MOF_RAC_example
    import MOF_prediction_models
    import final_prediction

def do_all(cif_filename, startpath = None):

    ### Initialise the ML models (train and validate)

    Temperature_Model = MOF_prediction_models.TT_Model(target = 'temperature', target_unit = '°C')
    Temperature_Model.train()
    Temperature_Model.validate()

    Time_Model = MOF_prediction_models.TT_Model(target = 'time', target_unit = 'h')
    Time_Model.train()
    Time_Model.validate()

    Additive_Model = MOF_prediction_models.Additive_Model(target = 'additive', feature_names = ['additive_category'])
    Additive_Model.train()
    Additive_Model.validate()

    Solvent_Model = MOF_prediction_models.Solvent_Model(target = 'solvent', feature_names = ['param%i'%(i+1) for i in range(5)])
    Solvent_Model.train()
    Solvent_Model.validate()

    ### Predict synthesis conditions if a cif file was given

    if cif_filename:
        if startpath is None:
            startpath = os.getcwd()

        ### Get a random name for the MOF
        now = datetime.now()
        date_string = now.strftime("%Y_%m_%d_%H_%M_%S")
        random_string = uuid.uuid4()
        MOF_random_name = "%s_%s"%(date_string, random_string)

        ### Log everything
        logfile = open("%s/mof_predictions.log"%(startpath), "a")
        logfile.write("%s %s\n"%(MOF_random_name, cif_filename))
        logfile.close()

        ### Download the uploaded cif file by the user and keep it in the following directory named cif
        if not os.path.exists("%s/cif_files/"%(startpath)):
            os.makedirs("%s/cif_files/"%(startpath))

        if os.path.exists("%s"%(cif_filename)):
            if not os.path.exists("%s/cif_files/%s"%(startpath, MOF_random_name)):
                os.makedirs("%s/cif_files/%s"%(startpath, MOF_random_name))
            ### Copy the downloaded cif file to the cif directory
            shutil.copy(cif_filename, f'{startpath}/cif_files/{MOF_random_name}/mof.cif')
        else:
            print("ERROR: file not found: %s"%(cif_filename))
            return(None, None, None, None)

        ### Calculation of RAC features of the downloaded cif file using Molsimplify
        df_new = MOF_RAC_example.compute_features(MOF_random_name, startpath)
        # df_new = pd.read_csv('additional_data/full_featurization_frame.csv')

        ### Predictions
        ML_models = [Temperature_Model, Time_Model, Additive_Model, Solvent_Model]
        for model in ML_models:
            model.make_predictions(MOF_random_name, df_new)

        temperature = Temperature_Model.get_final_prediction()
        time = Time_Model.get_final_prediction()
        additive = Additive_Model.get_final_prediction()
        solvent = Solvent_Model.get_final_prediction()

        # predictions_temperature = model_temperature.model_temperature(MOF_random_name, df_new, startpath)
        # predictions_time = model_time.model_time(MOF_random_name, df_new, startpath)
        # predictions_additive = model_additive.model_additive(MOF_random_name, df_new, startpath)
        # predictions_solvent = model_solvent.model_solvent(MOF_random_name, df_new, startpath)

        return(temperature, time, solvent, additive)

    else:
        return(None, None, None, None)

if __name__ == "__main__":

    if len(sys.argv) > 1: 
        cif_filename = sys.argv[1]
    else:
        cif_filename = None

    temperature, time, solvent, additive_string = do_all(cif_filename)

