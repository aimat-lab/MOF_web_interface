#First, We import the python libraries necesary to run this calculation

###########Standard_Python_Libraries#######################
import os
import sys
import numpy as np
import random
import joblib
import time
from datetime import datetime
import uuid
from matplotlib import pyplot as plt

##############rdkit_library##########################
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
#########panda to deal with csv files##############
import pandas as pd

###########sklearn_libary for ML models###################
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold

#########yaml library to deal with "settings.yml"(containing various parameters) file#############
import yaml






#######Defining the mean absolute error(mae) and r2 as an output of a function#########


def reg_stats(y_true,y_pred,scaler=None):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if scaler:
        y_true_unscaled = scaler.inverse_transform(y_true)
        y_pred_unscaled = scaler.inverse_transform(y_pred)
    r2 = sklearn.metrics.r2_score(y_true,y_pred)
    mae = sklearn.metrics.mean_absolute_error(y_true_unscaled, y_pred_unscaled)
    return r2,mae



def model_additive(MOF_random_name = None, df_new = None, startpath = None):

    if startpath is None:
        startpath = os.getcwd()

    if MOF_random_name is None:
        now = datetime.now()
        date_string = now.strftime("%Y_%m_%d_%H_%M_%S")
        random_string = uuid.uuid4()
        MOF_random_name = "%s_%s"%(date_string, random_string)


    #########seed########################################

    np.random.seed(1)
    random.seed(1)


    #############Print the Versions of the Sklearn and rdkit library#####################

    print("   ###   Libraries:")
    print('   ---   sklearn:{}'.format(sklearn.__version__))
    print('   ---   rdkit:{}'.format(rdkit.__version__))

    features_basic=["f-chi-0-all","f-chi-1-all","f-chi-2-all","f-chi-3-all","f-Z-0-all","f-Z-1-all","f-Z-2-all","f-Z-3-all","f-I-0-all","f-I-1-all","f-I-2-all","f-I-3-all","f-T-0-all","f-T-1-all","f-T-2-all","f-T-3-all","f-S-0-all","f-S-1-all","f-S-2-all","f-S-3-all","mc-chi-0-all","mc-chi-1-all","mc-chi-2-all","mc-chi-3-all","mc-Z-0-all","mc-Z-1-all","mc-Z-2-all","mc-Z-3-all","mc-I-0-all","mc-I-1-all","mc-I-2-all","mc-I-3-all","mc-T-0-all","mc-T-1-all","mc-T-2-all","mc-T-3-all","mc-S-0-all","mc-S-1-all","mc-S-2-all","mc-S-3-all","D_mc-chi-0-all","D_mc-chi-1-all","D_mc-chi-2-all","D_mc-chi-3-all","D_mc-Z-0-all","D_mc-Z-1-all","D_mc-Z-2-all","D_mc-Z-3-all","D_mc-I-0-all","D_mc-I-1-all","D_mc-I-2-all","D_mc-I-3-all","D_mc-T-0-all","D_mc-T-1-all","D_mc-T-2-all","D_mc-T-3-all","D_mc-S-0-all","D_mc-S-1-all","D_mc-S-2-all","D_mc-S-3-all","f-lig-chi-0","f-lig-chi-1","f-lig-chi-2","f-lig-chi-3","f-lig-Z-0","f-lig-Z-1","f-lig-Z-2","f-lig-Z-3","f-lig-I-0","f-lig-I-1","f-lig-I-2","f-lig-I-3","f-lig-T-0","f-lig-T-1","f-lig-T-2","f-lig-T-3","f-lig-S-0","f-lig-S-1","f-lig-S-2","f-lig-S-3","lc-chi-0-all","lc-chi-1-all","lc-chi-2-all","lc-chi-3-all","lc-Z-0-all","lc-Z-1-all","lc-Z-2-all","lc-Z-3-all","lc-I-0-all","lc-I-1-all","lc-I-2-all","lc-I-3-all","lc-T-0-all","lc-T-1-all","lc-T-2-all","lc-T-3-all","lc-S-0-all","lc-S-1-all","lc-S-2-all","lc-S-3-all","lc-alpha-0-all","lc-alpha-1-all","lc-alpha-2-all","lc-alpha-3-all","D_lc-chi-0-all","D_lc-chi-1-all","D_lc-chi-2-all","D_lc-chi-3-all","D_lc-Z-0-all","D_lc-Z-1-all","D_lc-Z-2-all","D_lc-Z-3-all","D_lc-I-0-all","D_lc-I-1-all","D_lc-I-2-all","D_lc-I-3-all","D_lc-T-0-all","D_lc-T-1-all","D_lc-T-2-all","D_lc-T-3-all","D_lc-S-0-all","D_lc-S-1-all","D_lc-S-2-all","D_lc-S-3-all","D_lc-alpha-0-all","D_lc-alpha-1-all","D_lc-alpha-2-all","D_lc-alpha-3-all","func-chi-0-all","func-chi-1-all","func-chi-2-all","func-chi-3-all","func-Z-0-all","func-Z-1-all","func-Z-2-all","func-Z-3-all","func-I-0-all","func-I-1-all","func-I-2-all","func-I-3-all","func-T-0-all","func-T-1-all","func-T-2-all","func-T-3-all","func-S-0-all","func-S-1-all","func-S-2-all","func-S-3-all","func-alpha-0-all","func-alpha-1-all","func-alpha-2-all","func-alpha-3-all","D_func-chi-0-all","D_func-chi-1-all","D_func-chi-2-all","D_func-chi-3-all","D_func-Z-0-all","D_func-Z-1-all","D_func-Z-2-all","D_func-Z-3-all","D_func-I-0-all","D_func-I-1-all","D_func-I-2-all","D_func-I-3-all","D_func-T-0-all","D_func-T-1-all","D_func-T-2-all","D_func-T-3-all","D_func-S-0-all","D_func-S-1-all","D_func-S-2-all","D_func-S-3-all","D_func-alpha-0-all","D_func-alpha-1-all","D_func-alpha-2-all","D_func-alpha-3-all"] 

    target="additive_category"  #Here output of the ML classification model is additive category

    csvfilename="%s/datasets/rac_features_additive.csv"%(startpath)
    df = pd.read_csv(csvfilename)  ## The csv file containing the input output of the ML models

    if df_new is None and os.path.exists('%s/full_featurization_frame.csv'%(startpath)):
        print("WARNING: USING THE TEST FEATURES FROM additional_data/full_featurization_frame.csv")
        df_new = pd.read_csv('%s/additional_data/full_featurization_frame.csv'%(startpath))



    ###########Open_A_file_to_write_all_prediction###
    if not os.path.exists("%s/predictions"%(outdir)):
        os.makedirs("%s/predictions"%(outdir))
    single_prediction=open('%s/predictions/%s_additive_prediction.dat'%(startpath, MOF_random_name),'w')

    ###########Training of ML Model given the input ("df") and output("target") of the model############### 



    fontname='Arial' 
    outdir = startpath

    print("start training")

    if not os.path.exists("%s/models"%(outdir)):
        os.makedirs("%s/models"%(outdir))

    if not os.path.exists("%s/prediction_data"%(outdir)):
        os.makedirs("%s/prediction_data"%(outdir))

    if not os.path.exists("%s/prediction_data/scatter_plots_additive"%(outdir)):
        os.makedirs("%s/prediction_data/scatter_plots_additive"%(outdir))

    if not os.path.exists("%s/models/models_additive"%(outdir)):
        os.makedirs("%s/models/models_additive"%(outdir))

    if not os.path.exists("%s/prediction_data/predictions_rawdata_additive"%(outdir)):
        os.makedirs("%s/prediction_data/predictions_rawdata_additive"%(outdir))

    # new sample
    x_new_unscaled=df_new[features_basic].values

    ##### dividing the full data in 10 different train and test set############

    X = np.array(df.index.tolist())
    kf = KFold(n_splits=10,shuffle=True)
    kf.get_n_splits(X)
    counter=0

    single_predictions = []
    #now train ML model over all these 10 different train test split###########
    for train_index, test_index in kf.split(X):
        counter=counter+1


        if not os.path.exists("%s/models/models_additive/random_forest_regression_%i.joblib"%(outdir, counter)):
        
            y=df[target].values.reshape(-1,1)  
            y_train, y_test = y[train_index], y[test_index]
        
            ###Preparing the input according to precomputed features by KULIK
            ### and co-workers. All the features are loaded within features_basic array
          
            
        
            ### standard scaling of the input features
            x_scaler_feat = StandardScaler()
            x_unscaled_feat=df[features_basic].values



            #x_old_new=np.vstack([x_unscaled_feat,x_new_unscaled])

            x_scaled_feat=x_scaler_feat.fit_transform(x_unscaled_feat)
            #print(x_new_unscaled)
            #print(x_new_unscaled.shape)
            #print(x_old_new.shape,x_old_new_feat.shape)
            #x_feat=np.vsplit(x_old_new_feat, np.array([670, 1]))
            #print (len(x_feat))

            # transform the new datapoint
            x_new_scaled = x_scaler_feat.transform(x_new_unscaled).reshape(1, -1)

            print(x_new_scaled.shape)


            #x_feat = x_scaler_feat.fit_transform(x_unscaled_feat)
            #n_feat = len(features_basic)
            x = x_scaled_feat
            print (x.shape)

            ## Dividing the input in train and test data set
            x_train, x_test = x[train_index],x[test_index]


            #final training and test data dimensions are printed here
            print("   ---   Training and test data dimensions:")
            print(x_train.shape,x_test.shape,y_train.shape, y_test.shape)


         

            ############################
            # RandomForestClassifier model is initiated now
            ############################
            model =  RandomForestClassifier(class_weight="balanced",max_depth=5)  
            
            #fit the model
            model.fit(x_train,y_train.ravel())

            joblib.dump(model, "%s/models/models_additive/random_forest_regression_%i.joblib"%(outdir, counter))
            joblib.dump(x_scaler_feat, "%s/models/models_additive/random_forest_regression_%i_x_scaler.joblib"%(outdir, counter))
        
             
            print("\n   ###   RandomForestClassifier:")

            ####Evaluation of the performance of the fitted model
            ####over training and test data set
            y_pred_train = model.predict(x_train)
            y_pred_test = model.predict(x_test)
            np.savetxt("%s/prediction_data/predictions_rawdata_additive/y_real_"%(outdir)+str(counter)+"_test.txt", y_test)
            np.savetxt("%s/prediction_data/predictions_rawdata_additive/y_real_"%(outdir)+str(counter)+"_train.txt", y_train)
            np.savetxt("%s/prediction_data/predictions_rawdata_additive/y_RFR_"%(outdir)+str(counter)+"_test.txt", y_pred_test)
            np.savetxt("%s/prediction_data/predictions_rawdata_additive/y_RFR_"%(outdir)+str(counter)+"_train.txt", y_pred_train)

        else:
            model = joblib.load("%s/models/models_additive/random_forest_regression_%i.joblib"%(outdir, counter))
            x_scaler_feat = joblib.load("%s/models/models_additive/random_forest_regression_%i_x_scaler.joblib"%(outdir, counter))
            x_new_scaled = x_scaler_feat.transform(x_new_unscaled).reshape(1, -1)


        y_new_pred=model.predict(x_new_scaled)
        print ("HEREEEEEE IS YOUR PREDICTION:" ,y_new_pred[0])
        single_prediction.write(str(y_new_pred[0])+'\n')
        single_predictions.append(y_new_pred[0])

    single_prediction.close()

    return(single_predictions)



if __name__ == "__main__":
    model_additive()




