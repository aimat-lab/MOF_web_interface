#First, We import the python libraries necesary to run this calculation

###########Standard_Python_Libraries#######################
import os
import sys
import csv
import glob
import numpy as np
import random
import joblib
import math
import time
from datetime import datetime
import uuid
from matplotlib import pyplot as plt

##############rdkit_library##########################
# import rdkit
# from rdkit import Chem
# from rdkit.Chem import AllChem
# from rdkit.Chem import Draw

#########panda to deal with csv files##############
import pandas as pd

###########sklearn_libary for ML models###################
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import KFold, train_test_split

#########yaml library to deal with "settings.yml"(containing various parameters) file#############
import yaml

### Set global variables ###

features_basic=["f-chi-0-all","f-chi-1-all","f-chi-2-all","f-chi-3-all","f-Z-0-all","f-Z-1-all","f-Z-2-all","f-Z-3-all","f-I-0-all","f-I-1-all","f-I-2-all","f-I-3-all","f-T-0-all","f-T-1-all","f-T-2-all","f-T-3-all","f-S-0-all","f-S-1-all","f-S-2-all","f-S-3-all","mc-chi-0-all","mc-chi-1-all","mc-chi-2-all","mc-chi-3-all","mc-Z-0-all","mc-Z-1-all","mc-Z-2-all","mc-Z-3-all","mc-I-0-all","mc-I-1-all","mc-I-2-all","mc-I-3-all","mc-T-0-all","mc-T-1-all","mc-T-2-all","mc-T-3-all","mc-S-0-all","mc-S-1-all","mc-S-2-all","mc-S-3-all","D_mc-chi-0-all","D_mc-chi-1-all","D_mc-chi-2-all","D_mc-chi-3-all","D_mc-Z-0-all","D_mc-Z-1-all","D_mc-Z-2-all","D_mc-Z-3-all","D_mc-I-0-all","D_mc-I-1-all","D_mc-I-2-all","D_mc-I-3-all","D_mc-T-0-all","D_mc-T-1-all","D_mc-T-2-all","D_mc-T-3-all","D_mc-S-0-all","D_mc-S-1-all","D_mc-S-2-all","D_mc-S-3-all","f-lig-chi-0","f-lig-chi-1","f-lig-chi-2","f-lig-chi-3","f-lig-Z-0","f-lig-Z-1","f-lig-Z-2","f-lig-Z-3","f-lig-I-0","f-lig-I-1","f-lig-I-2","f-lig-I-3","f-lig-T-0","f-lig-T-1","f-lig-T-2","f-lig-T-3","f-lig-S-0","f-lig-S-1","f-lig-S-2","f-lig-S-3","lc-chi-0-all","lc-chi-1-all","lc-chi-2-all","lc-chi-3-all","lc-Z-0-all","lc-Z-1-all","lc-Z-2-all","lc-Z-3-all","lc-I-0-all","lc-I-1-all","lc-I-2-all","lc-I-3-all","lc-T-0-all","lc-T-1-all","lc-T-2-all","lc-T-3-all","lc-S-0-all","lc-S-1-all","lc-S-2-all","lc-S-3-all","lc-alpha-0-all","lc-alpha-1-all","lc-alpha-2-all","lc-alpha-3-all","D_lc-chi-0-all","D_lc-chi-1-all","D_lc-chi-2-all","D_lc-chi-3-all","D_lc-Z-0-all","D_lc-Z-1-all","D_lc-Z-2-all","D_lc-Z-3-all","D_lc-I-0-all","D_lc-I-1-all","D_lc-I-2-all","D_lc-I-3-all","D_lc-T-0-all","D_lc-T-1-all","D_lc-T-2-all","D_lc-T-3-all","D_lc-S-0-all","D_lc-S-1-all","D_lc-S-2-all","D_lc-S-3-all","D_lc-alpha-0-all","D_lc-alpha-1-all","D_lc-alpha-2-all","D_lc-alpha-3-all","func-chi-0-all","func-chi-1-all","func-chi-2-all","func-chi-3-all","func-Z-0-all","func-Z-1-all","func-Z-2-all","func-Z-3-all","func-I-0-all","func-I-1-all","func-I-2-all","func-I-3-all","func-T-0-all","func-T-1-all","func-T-2-all","func-T-3-all","func-S-0-all","func-S-1-all","func-S-2-all","func-S-3-all","func-alpha-0-all","func-alpha-1-all","func-alpha-2-all","func-alpha-3-all","D_func-chi-0-all","D_func-chi-1-all","D_func-chi-2-all","D_func-chi-3-all","D_func-Z-0-all","D_func-Z-1-all","D_func-Z-2-all","D_func-Z-3-all","D_func-I-0-all","D_func-I-1-all","D_func-I-2-all","D_func-I-3-all","D_func-T-0-all","D_func-T-1-all","D_func-T-2-all","D_func-T-3-all","D_func-S-0-all","D_func-S-1-all","D_func-S-2-all","D_func-S-3-all","D_func-alpha-0-all","D_func-alpha-1-all","D_func-alpha-2-all","D_func-alpha-3-all"]

### Create global directories ###

startpath = os.getcwd()

models_path = '%s/models'%startpath
validation_path = '%s/validation'%startpath
predictions_path = '%s/predictions'%startpath

for dirname in [models_path, validation_path, predictions_path]:
    if not os.path.exists(dirname):
        os.makedirs(dirname)

### Define the mean absolute error(mae), root mean squared error (rmse)  and r2 as an output of a function ###

def reg_stats(y_true, y_pred, y_scaler = None):

    r2 = sklearn.metrics.r2_score(y_true,y_pred)

    if y_scaler:
        y_true_unscaled = y_scaler.inverse_transform(y_true)
        y_pred_unscaled = y_scaler.inverse_transform(y_pred)

        mae = sklearn.metrics.mean_absolute_error(y_true_unscaled, y_pred_unscaled)
        rmse = np.sqrt(sklearn.metrics.mean_squared_error(y_true_unscaled, y_pred_unscaled))

    else:
        mae, rmse = None, None

    return r2, mae, rmse

def make_histogram(data, bin_width, xlabel, filename):

        fig, ax = plt.subplots()

        bin_width = 0.5
        min_bin = math.floor(2*min(data)/bin_width)*bin_width/2
        max_bin = math.ceil(2*max(data)/bin_width)*bin_width/2

        ax.set_xlim(xmin = min_bin-bin_width, xmax = max_bin+bin_width)

        n, bins, patches = ax.hist(data,bins=np.arange((math.ceil(min_bin/bin_width)-0.5)*bin_width,(1.5+math.floor(max_bin/bin_width))*bin_width,bin_width))

        plt.ylabel('Count')
        plt.xlabel(xlabel)
        plt.savefig(filename,dpi=300)
        plt.close()

class RF_Model():

    # ### Print the version of the sklearn library ###
    #
    # print("   ###   Libraries used in model_1d_rfr:")
    # print('   ---   sklearn:{}'.format(sklearn.__version__))
    # print('   ---   rdkit:{}'.format(rdkit.__version__))

    def __init__(self, target, target_unit = '', feature_names = None):

        if not feature_names:
            feature_names = [target]

        self.target = target
        self.target_unit = target_unit
        self.feature_names = feature_names
        self.n_feat = len(feature_names)

        ### Seed ###

        np.random.seed(1)
        random.seed(1)
        
        ### Create directories for target ###
    
        self.models_target_path = '%s/models_%s'%(models_path, target)
        self.validation_target_path = '%s/validation_%s'%(validation_path, target)
        # self.scatter_plots_path = '%s/scatter_plots_%s'%(models_path,target)

        for dirname in [self.models_target_path, self.validation_target_path]:
            if not os.path.exists(dirname):
                os.makedirs(dirname)

        # Prepare data for the ML model according to precomputed features by Kulik et. al. with given input(df) and output("target") #
        self.df = pd.read_csv('%s/datasets/rac_features_%s.csv'%(startpath,target))
        self.all_indices = np.array(self.df.index.tolist())

        # definition and standard scaling of the input features
        self.x_scaler_feat = StandardScaler()
        x_unscaled_feat = self.df[features_basic].values
        self.x = self.x_scaler_feat.fit_transform(x_unscaled_feat)

        # save x_scaler (is it needed somewhere later?)
        # joblib.dump(x_scaler_feat, '%s/random_forest_%s_x_scaler.joblib'%(self.models_target_path, self.model_type))

        # definition of the output of the ML model 
        self.y_unscaled = self.df[feature_names].values

    def train(self, base_path = None, rs = None):
        
        if not base_path:
            base_path = self.models_target_path

        kf = KFold(n_splits = 10, shuffle = True, random_state = rs)

        # Divide the full data in 10 sets with each 10% test and 90% train/validation set
        count_ext = 1
        for train_val_idx, test_idx in kf.split(self.all_indices):
        
            x_test, y_test = self.x[test_idx], self.y[test_idx]
    
            # train ML model over all these 10 different train validation split #
            count_int = 1
            for train_idx, val_idx in kf.split(train_val_idx):
                train_idx = np.array([train_val_idx[i] for i in train_idx])
                val_idx = np.array([train_val_idx[i] for i in val_idx])

                RFR_basepath = '%s/random_forest_%s_model_%02i_%02i'%(base_path, self.model_type, count_ext, count_int)
                if not os.path.exists('%s.joblib'%RFR_basepath):

                    print("\nStart training of %s model %2i for cross validation set %2i"%(self.target,count_int, count_ext))

                    # Divide the output in train and validation data
                    y_train, y_val = self.y[train_idx], self.y[val_idx]
        
                    # Divide the input in train and validation data set
                    x_train, x_val = self.x[train_idx], self.x[val_idx]

                    # Print final training and validation data dimensions 
                    print("   ---   Training and validation data dimensions:")
                    print(x_train.shape,x_val.shape,y_train.shape, y_val.shape)

                    # Initiate and fit the RandomForestRegressor model 
                    
                    model = self.rf_model

                    if self.n_feat == 1:
                        model.fit(x_train,y_train.ravel())
                    else:
                        model.fit(x_train,y_train)

                    # save the model
                    joblib.dump(model, '%s.joblib'%RFR_basepath)

                    # evaluate the performance of the fitted model over training and validation data set

                    y_pred_train = model.predict(x_train).reshape(-1, self.n_feat)
                    y_pred_val = model.predict(x_val).reshape(-1, self.n_feat)
                    y_pred_test = model.predict(x_test).reshape(-1, self.n_feat)

                    if self.regression:
                        print("\n   ###   RandomForestRegressor:")
                        if self.n_feat == 1:
                            r2_GBR_train, mae_GBR_train, rmse_GBR_train = reg_stats(y_train, y_pred_train, self.y_scaler)
                            print("   ---   Training (r2, MAE, RMSE):   %.3f %.3f %.3f"%(r2_GBR_train, mae_GBR_train, rmse_GBR_train))
                            r2_GBR_val, mae_GBR_val, rmse_GBR_val = reg_stats(y_val, y_pred_val, self.y_scaler)
                            print("   ---   Validating (r2, MAE, RMSE): %.3f %.3f %.3f"%(r2_GBR_val, mae_GBR_val, rmse_GBR_val))
                            r2_GBR_test, mae_GBR_test, rmse_GBR_test = reg_stats(y_test, y_pred_test, self.y_scaler)
                            print("   ---   Testing (r2, MAE, RMSE):    %.3f %.3f %.3f"%(r2_GBR_test, mae_GBR_test, rmse_GBR_test))
                        else: 
                            r2_GBR_train, mae_GBR_train, rmse_GBR_train = reg_stats(y_train, y_pred_train)
                            print("   ---   Training (r2, MAE, RMSE):   %.3f"%r2_GBR_train)
                            r2_GBR_val, mae_GBR_val, rmse_GBR_val = reg_stats(y_val, y_pred_val)
                            print("   ---   Validating (r2, MAE, RMSE): %.3f"%r2_GBR_val)
                            r2_GBR_test, mae_GBR_test, rmse_GBR_test = reg_stats(y_test, y_pred_test)
                            print("   ---   Testing (r2, MAE, RMSE):    %.3f"%r2_GBR_test)

                    ### Adapt scatter plots to regression / classification

                    # scale back the output

                    # y_val_unscaled = y_scaler.inverse_transform(y_val)
                    # y_train_unscaled = y_scaler.inverse_transform(y_train)
                    # y_pred_val_unscaled = y_scaler.inverse_transform(y_pred_val)
                    # y_pred_train_unscaled = y_scaler.inverse_transform(y_pred_train)

                    # save and plot of the predictions

                    # plt.figure()
                    # plt.scatter(y_pred_train_unscaled, y_train_unscaled, marker="o", c="C1", label="Training: r$^2$ = %.3f"%(r2_GBR_train))
                    # plt.scatter(y_pred_val_unscaled, y_val_unscaled, marker="o", c="C2", label="Testing: r$^2$ = %.3f"%(r2_GBR_val))
                    # plt.scatter(y_pred_train_unscaled, y_train_unscaled, marker="o", c="C1", label="Training: MAE = %.3f"%(mae_GBR_train))
                    # plt.scatter(y_pred_val_unscaled, y_val_unscaled, marker="o", c="C2", label="Testing: MAE = %.3f"%(mae_GBR_val))
                    # plt.plot(y_train_unscaled,y_train_unscaled)
                    # plt.title('RandomForestRegressor')

                    # plt.ylabel("Experimental %s [%s]"%(target.capitalize(),target_unit))
                    # plt.xlabel("Predicted %s [%s]"%(target.capitalize(),target_unit))
                    # plt.legend(loc="upper left")
                    # plt.savefig('%s/full_data_RFR_%02i_%02i.png'%(self.scatter_plots_path, count_ext, count_int), dpi=300)
                    # plt.close()
            
                count_int +=1
        
            count_ext +=1

    def validate(self):
        for rs in [453, 7644, 24369, 42548, 86310, 273214, 358412, 414551, 712111, 983187]:

            models_rs_path = '/%s/models_rs_%06i'%(self.validation_target_path, rs)
            if not os.path.exists(models_rs_path):
                os.makedirs(models_rs_path)

            self.train(base_path = models_rs_path, rs = rs)

            rs_csv_file = '%s/std_list_rs_%06i_split_RFR.csv'%(self.validation_target_path, rs)
            rs_hist_file = '%s/std_histogram_rs_%06i_split_RFR.png'%(self.validation_target_path, rs)

            if not os.path.exists(rs_csv_file) or not os.path.exists(rs_hist_file):

                kf = KFold(n_splits = 10, shuffle = True, random_state = rs)
                y_pred_test_all = []
                count_ext = 1

                for train_val_idx, test_idx in kf.split(self.all_indices):
                    x_test = self.x[test_idx]
                    y_pred_test_set = []
                    count_int = 1

                    for train_idx, val_idx in kf.split(train_val_idx):
                        train_idx = np.array([train_val_idx[i] for i in train_idx])
                        val_idx = np.array([train_val_idx[i] for i in val_idx])

                        y_val = self.y[val_idx]
                        x_val = self.x[val_idx]

                        # print("\nLoad %s model %2i for cross validation set %2i"%(target, count_int, count_ext))
                        model = joblib.load('%s/random_forest_%s_model_%02i_%02i.joblib'%(models_rs_path, self.model_type, count_ext, count_int))
        
                        # prediction for independent test split
                        y_pred_test_scaled = model.predict(x_test).reshape(-1, self.n_feat)
                        y_pred_test_unscaled = self.y_scaler.inverse_transform(y_pred_test_scaled)
                        y_pred_test_set.append(y_pred_test_unscaled)

                        count_int += 1

                    y_pred_test_all.append([test_idx, y_pred_test_set])

                    count_ext += 1

                make_validation_histogram(y_pred_test_all, rs_csv_file, rs_hist_file) # define method in subclass depending on property

        scatter_plot_file = '%s/std_scatter_plots_10_splits_%s.png'%(self.validation_target_path, self.target)

        if not os.path.exists(scatter_plot_file):

            csv_files = glob.glob('%s/std_list_rs*.csv'%validation_target_path)    

            with open(csv_files[0]) as infile:
                csv_lines=infile.readlines()[1:]
                names=[line.split(',')[0] for line in csv_lines]
                error_values=[[float(line.split(',')[1].rstrip()) for line in  csv_lines]]
        
            for csv_file in csv_files[1:]:
                with open(csv_file) as infile:
                    csv_lines=infile.readlines()[1:]
                    error_values.append([float(line.split(',')[1].rstrip()) for line in  csv_lines])

            xval = range(len(names))

            plt.figure()
            plt.title('Standard deviation of different KFold splits for all structures')
            plt.ylabel("Std")
            plt.xlabel("Structure no.")

            for i in range(len(error_values)):
                yval = [error_values[i][j] for j in range(len(xval))]
                plt.scatter(xval,yval, marker="o",s=6)

            plt.savefig(scatter_plot_file, dpi=300)

    def make_predictions(self, MOF_random_name, df_new):

        single_predictions = []

        x_new_unscaled = df_new[features_basic].values
        x_new_scaled = self.x_scaler_feat.transform(x_new_unscaled).reshape(1, -1)

        kf = KFold(n_splits = 10, shuffle = True, random_state = None)

        count_ext = 1
        for train_val_idx, test_idx in kf.split(self.all_indices):
            single_prediction_set = []

            count_int = 1
            for train_idx, val_idx in kf.split(train_val_idx):
                train_idx = np.array([train_val_idx[i] for i in train_idx])
                val_idx = np.array([train_val_idx[i] for i in val_idx])

                RFR_basepath = '%s/random_forest_%s_model_%02i_%02i'%(self.models_target_path, self.model_type, count_ext, count_int)

                model = joblib.load('%s.joblib'%RFR_basepath)
                y_new_pred = model.predict(x_new_scaled).reshape(-1, self.n_feat)
                
                if self.regression:
                    y_new_pred_unscaled = self.y_scaler.inverse_transform(y_new_pred)
                    single_prediction_set.append(y_new_pred_unscaled[0])
                else:
                    single_prediction_set.append(y_new_pred[0])

                count_int += 1

            single_predictions.append(single_prediction_set)
            
            count_ext += 1

        self.single_predictions = np.array(single_predictions)

        # Write all predictions to a file
        # np.savetxt('%s/%s_%s_prediction.dat'%(predictions_path, MOF_random_name, self.target), self.single_predictions)

class Classification_Model(RF_Model):
    
    regression = False

    model_type='classification'
    rf_model = RandomForestClassifier(max_depth=5)

    def __init__(self, *args, **kwargs):
        super(Classification_Model,self).__init__(*args, **kwargs)
        
        self.y_scaler = None
        self.y = self.y_unscaled

class Regression_Model(RF_Model):
    
    regression = True

    model_type='regression'
    rf_model = RandomForestRegressor(max_depth=5)

    def __init__(self, *args, **kwargs):
        super(Regression_Model,self).__init__(*args, **kwargs)
        
        # standard scaling of the output
        self.y_scaler = StandardScaler()
        self.y = self.y_scaler.fit_transform(self.y_unscaled)

        # save y_scaler (is it needed somewhere later?)
        # joblib.dump(y_scaler, '%s/random_forest_%s_y_scaler.joblib'%(self.models_target_path, self.model_type))

class Additive_Model(Classification_Model):

    def get_final_prediction(self):

        additive_categories = {0:'Base', 1:'Neutral/no additive', 2:'Acid'}

        final_prediction = np.bincount(self.single_predictions.ravel()).argmax()
        return_string = additive_categories[final_prediction]
        print ('ML Predicted %s: %s'%(self.target.capitalize(), return_string))

        return(return_string)

    def make_validation_histogram(y_pred_test_all, csv_file, hist_file):

        correct_pred, filenames = [], []

        for y_pred_set in y_pred_test_all:
            y_pred_struct = [[int(y_pred_set[1][j][i]) for j in range(len(y_pred_set[1]))] for i in range(len(y_pred_set[1][0]))]
            correct_pred += [y.count(max(y,key=y.count))/len(y)for y in y_pred_struct]
            filenames += [df['filename'].iloc[idx] for idx in y_pred_set[0]]

        filenames, correct_pred = zip(*sorted(zip(filenames, std)))

        with open(csv_file,'w') as outfile:
            outfile.write('filename, correct predictions\n')
            for i in range(len(correct_pred)):
                outfile.write('%s, %s\n'%(filenames[i], correct_pred[i]))

        make_histogram(correct_pred, 0.1, 'Occurence of most frequent prediction in the %s models'%self.target, hist_file)

class Solvent_Model(Regression_Model):

    def get_final_prediction(self):
        solvent_names = pd.read_csv("%s/additional_data/local_solvent_full.csv"%(startpath))['solvent_name']
        solvent_data = np.loadtxt('%s/additional_data/scaled_five_parameter_local_solvent.dat'%startpath)

        centroid = [np.average([single_prediction[i] for single_prediction in self.single_predictions.reshape(-1, self.n_feat)]) for i in range(self.n_feat)]
        centroid_distances = [np.sqrt(sum([(centroid[j]-solvent_data[i][j])**2 for j in range(len(centroid))])) for i in range(len(solvent_data))]

        solvent_order = np.argsort(centroid_distances)
        print('ML Predicted Best 5 %ss: %s'%(self.target.capitalize(),', '.join([solvent_names[solvent_order[i]] for i in range(5)])))

        return(solvent_names[solvent_order[0]])

    def make_validation_histogram(y_pred_test_all, csv_file, hist_file):

        centroid_dist, filenames = [], []
                
        for y_pred_set in y_pred_test_all:
            centroids = [[np.average([y_pred_set[1][k][i][j] for k in range(len(y_pred_set[1]))]) for j in range(len(y_pred_set[1][0][0]))] for i in range(len(y_pred_set[1][0]))]
            centroid_dist += [np.sqrt(sum([(y_pred_set[1][k][i][j]-centroids[i][j])**2 for j in range(len(centroids[0])) for k in range(len(y_pred_set[1]))])/len(y_pred_set[1])) for i in range(len(centroids))]

            filenames += [df['filename'].iloc[idx] for idx in y_pred_set[0]]

        filenames, std = zip(*sorted(zip(filenames, centroid_dist)))

        with open(rs_csv_file,'w') as outfile:
            outfile.write('filename, centroid distance\n')
            for i in range(len(centroid_dist)):
                outfile.write('%s, %s\n'%(filenames[i], centroid_dist[i]))

                
        make_histogram(centroid_dist, 0.05, 'Distances from centroid of the scaled %s models'%self.target, hist_file)

class TT_Model(Regression_Model):

    def get_final_prediction(self):

        final_prediction = int(np.rint(np.average(self.single_predictions)))
        return_string = str(final_prediction)
        print ('ML Predicted %s: %3i %s'%(self.target.capitalize(), final_prediction, self.target_unit))

        return(return_string)

    
    def make_validation_histogram(y_pred_test_all, csv_file, hist_file):
        
        std, filenames = [], []

        for y_pred_set in y_pred_test_all:
            std += [np.std([y_pred_set[1][j][i] for j in range(len(y_pred_set[1]))]) for i in range(len(y_pred_set[1][0]))]
            filenames += [self.df['filename'].iloc[idx] for idx in y_pred_set[0]]

        filenames, std = zip(*sorted(zip(filenames, std)))

        with open(csv_file,'w') as outfile:
            outfile.write('filename, std\n')
            for i in range(len(std)):
                outfile.write('%s, %s\n'%(filenames[i], std[i]))

        make_histogram(std, 0.5, 'Standard deviation of the %s models'%self.target, hist_file)



