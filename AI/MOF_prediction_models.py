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

models_base_path = '%s/models'%startpath
validation_path = '%s/validation'%startpath
predictions_path = '%s/predictions'%startpath

for dirname in [models_base_path, validation_path, predictions_path]:
    if not os.path.exists(dirname):
        os.makedirs(dirname)

kf = KFold(n_splits = 10, shuffle = True)

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

    def __init__(self):

        self.n_feat = len(self.feature_names)

        ### Seed ###

        np.random.seed(1)
        random.seed(1)
        
        ### Create directories for target ###
    
        self.models_target_path = '%s/models_%s'%(models_base_path, self.target)
        self.validation_target_path = '%s/validation_%s'%(validation_path, self.target)
        # self.scatter_plots_path = '%s/scatter_plots_%s'%(models_base_path, self.target)

        for dirname in [self.models_target_path, self.validation_target_path]:
            if not os.path.exists(dirname):
                os.makedirs(dirname)

        # Prepare data for the ML model according to precomputed features by Kulik et. al. with given input(df) and output("target") #
        self.df = pd.read_csv('%s/datasets/rac_features_%s.csv'%(startpath, self.target))
        self.all_indices = np.array(self.df.index.tolist())

        # definition and standard scaling of the input features
        self.x_scaler_feat = StandardScaler()
        x_unscaled_feat = self.df[features_basic].values
        self.x = self.x_scaler_feat.fit_transform(x_unscaled_feat)

        # save x_scaler (is it needed somewhere later?)
        # joblib.dump(x_scaler_feat, '%s/random_forest_%s_x_scaler.joblib'%(self.models_target_path, self.model_type))

        # definition of the output of the ML model 
        self.y_unscaled = self.df[self.feature_names].values

    def train_model(self, train_idx, val_idx, model_filename):

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
        joblib.dump(model, model_filename)

        # evaluate the performance of the fitted model over training and validation data set

        y_pred_train = model.predict(x_train).reshape(-1, self.n_feat)
        y_pred_val = model.predict(x_val).reshape(-1, self.n_feat)
        #y_pred_test = model.predict(x_test).reshape(-1, self.n_feat)

        if self.regression:
            print("\n   ###   RandomForestRegressor:")
            if self.n_feat == 1:
                r2_GBR_train, mae_GBR_train, rmse_GBR_train = reg_stats(y_train, y_pred_train, self.y_scaler)
                print("   ---   Training (r2, MAE, RMSE):   %.3f %.3f %.3f"%(r2_GBR_train, mae_GBR_train, rmse_GBR_train))
                r2_GBR_val, mae_GBR_val, rmse_GBR_val = reg_stats(y_val, y_pred_val, self.y_scaler)
                print("   ---   Validating (r2, MAE, RMSE): %.3f %.3f %.3f"%(r2_GBR_val, mae_GBR_val, rmse_GBR_val))
         #       r2_GBR_test, mae_GBR_test, rmse_GBR_test = reg_stats(y_test, y_pred_test, self.y_scaler)
         #       print("   ---   Testing (r2, MAE, RMSE):    %.3f %.3f %.3f"%(r2_GBR_test, mae_GBR_test, rmse_GBR_test))
            else: 
                r2_GBR_train, mae_GBR_train, rmse_GBR_train = reg_stats(y_train, y_pred_train)
                print("   ---   Training (r2, MAE, RMSE):   %.3f"%r2_GBR_train)
                r2_GBR_val, mae_GBR_val, rmse_GBR_val = reg_stats(y_val, y_pred_val)
                print("   ---   Validating (r2, MAE, RMSE): %.3f"%r2_GBR_val)
         #       r2_GBR_test, mae_GBR_test, rmse_GBR_test = reg_stats(y_test, y_pred_test)
         #       print("   ---   Testing (r2, MAE, RMSE):    %.3f"%r2_GBR_test)

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
                
    def validate(self):
        
        models_path = '%s/models/'%self.validation_target_path
        if not os.path.exists(models_path):
            os.makedirs(models_path)

        csv_file = '%s/error_list_RF%s.csv'%(self.validation_target_path,self.model_type[0].capitalize())
        histogram_file = '%s/error_histogram_RF%s.png'%(self.validation_target_path,self.model_type[0].capitalize())

        # Divide the full data in 10 sets with each 10% test and 90% train/validation set
        count_ext = 1
        for train_val_idx, test_idx in kf.split(self.all_indices):
        
            x_test, y_test = self.x[test_idx], self.y[test_idx]
    
            # train ML model over all these 10 different train validation split #
            count_int = 1
            for train_idx, val_idx in kf.split(train_val_idx):
                train_idx = np.array([train_val_idx[i] for i in train_idx])
                val_idx = np.array([train_val_idx[i] for i in val_idx])

                model_filename = '%s/random_forest_%s_model_%02i_%02i.joblib'%(models_path, self.model_type, count_ext, count_int)

                if not os.path.exists(model_filename):
                    print("\nStart training of %s validation model %2i for cross validation set %2i"%(self.target,count_int, count_ext))
                    self.train_model(train_idx, val_idx, model_filename)

                count_int += 1
        
            count_ext += 1

        if not os.path.exists(csv_file) or not os.path.exists(histogram_file):

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
                
                    model = joblib.load('%s/random_forest_%s_model_%02i_%02i.joblib'%(models_path, self.model_type, count_ext, count_int))

                    y_pred_test = model.predict(x_test).reshape(-1, self.n_feat)
                    
                    if self.regression:
                        y_pred_test_unscaled = self.y_scaler.inverse_transform(y_pred_test)
                        y_pred_test_set.append(y_pred_test_unscaled)
                    else:
                        y_pred_test_set.append(y_pred_test)

                    count_int += 1

                y_pred_test_all.append([test_idx, y_pred_test_set])

                count_ext += 1

            self.make_validation_histogram(y_pred_test_all, csv_file, histogram_file) # define method in subclass depending on property

    def make_predictions(self, MOF_random_name, df_new):

        single_predictions = []

        x_new_unscaled = df_new[features_basic].values
        x_new_scaled = self.x_scaler_feat.transform(x_new_unscaled).reshape(1, -1)

        count = 1
        for train_idx, val_idx in kf.split(self.all_indices):

            model_filename = '%s/random_forest_%s_model_%02i.joblib'%(self.models_target_path, self.model_type, count)

            if not os.path.exists(model_filename):
                print('\nStart training of %s prediction model %2i'%(self.target, count))
                self.train_model(train_idx, val_idx, model_filename)

            model = joblib.load(model_filename)

            y_new_pred = model.predict(x_new_scaled).reshape(-1, self.n_feat)
                
            if self.regression:
                y_new_pred_unscaled = self.y_scaler.inverse_transform(y_new_pred)
                single_predictions.append(y_new_pred_unscaled[0])
            else:
                single_predictions.append(y_new_pred[0])

            count += 1

        self.single_predictions = np.array(single_predictions)

        # Write all predictions to a file
        # np.savetxt('%s/%s_%s_prediction.dat'%(predictions_path, MOF_random_name, self.target), self.single_predictions)


class Classification_Model(RF_Model):
    
    def __init__(self, *args, **kwargs):

        self.regression = False
        self.model_type = 'classification'
        self.rf_model = RandomForestClassifier(max_depth=5)

        super().__init__(*args, **kwargs)
        
        # No scaling of the output
        self.y_scaler = None
        self.y = self.y_unscaled


class Regression_Model(RF_Model):
    
    def __init__(self, *args, **kwargs):

        self.regression = True
        self.model_type = 'regression'
        self.rf_model = RandomForestRegressor(max_depth=5)

        super().__init__(*args, **kwargs)
        
        # standard scaling of the output
        self.y_scaler = StandardScaler()
        self.y = self.y_scaler.fit_transform(self.y_unscaled)

        # save y_scaler (is it needed somewhere later?)
        # joblib.dump(y_scaler, '%s/random_forest_%s_y_scaler.joblib'%(self.models_target_path, self.model_type))


class Additive_Model(Classification_Model):

    def __init__(self, *args, **kwargs):

        self.target = 'additive'
        self.feature_names = ['additive_category']

        super().__init__(*args, **kwargs)

    def get_final_prediction(self):

        additive_categories = {0:'Base', 1:'Neutral/none', 2:'Acid'}

        final_prediction = np.bincount(self.single_predictions.ravel()).argmax()
        n_final_prediction = np.bincount(self.single_predictions.ravel())[final_prediction]
        
        if n_final_prediction >= 10:
            certainty = 'high'
        elif n_final_prediction >= 9:
            certainty = 'medium'
        else: 
            certainty = 'low'

        print ('ML Predicted %s: %s (%s certainty prediction)'%(self.target.capitalize(), additive_categories[final_prediction], certainty))

        return([additive_categories[final_prediction], certainty])

    def make_validation_histogram(self, y_pred_test_all, csv_file, hist_file):

        correct_pred, filenames = [], []

        for y_pred_set in y_pred_test_all:
            y_pred_struct = [[int(y_pred_set[1][j][i]) for j in range(len(y_pred_set[1]))] for i in range(len(y_pred_set[1][0]))]
            correct_pred += [y.count(max(y,key=y.count))/len(y)for y in y_pred_struct]
            filenames += [self.df['filename'].iloc[idx] for idx in y_pred_set[0]]

        filenames, correct_pred = zip(*sorted(zip(filenames, correct_pred)))
        
        with open(csv_file,'w') as outfile:
            outfile.write('filename, correct predictions\n')
            for i in range(len(correct_pred)):
                outfile.write('%s, %s\n'%(filenames[i], correct_pred[i]))

        make_histogram(correct_pred, 0.1, 'Occurence of most frequent prediction in the %s models'%self.target, hist_file)


class Solvent_Model(Regression_Model):

    def __init__(self, *args, **kwargs):

        self.target = 'solvent'
        self.feature_names = ['param%i'%(i+1) for i in range(5)]

        super().__init__(*args, **kwargs)

    def get_final_prediction(self):
        solvent_names = pd.read_csv("%s/additional_data/local_solvent_full.csv"%(startpath))['solvent_name']
        solvent_data = np.loadtxt('%s/additional_data/scaled_five_parameter_local_solvent.dat'%startpath)

        centroid = [np.average([self.single_predictions[j][i] for j in range(len(self.single_predictions))]) for i in range(self.n_feat)]
        centroid_std = np.sqrt(sum([(self.single_predictions[i][j]-centroid[j])**2 for j in range(self.n_feat) for i in range(len(self.single_predictions))]))/len(self.single_predictions)
        centroid_distances = [np.sqrt(sum([(centroid[j]-solvent_data[i][j])**2 for j in range(len(centroid))])) for i in range(len(solvent_data))]
        solvent_order = np.argsort(centroid_distances)

        if centroid_std <= 0.05:
            certainty = 'high'
        elif centroid_std <= 0.1:
            certainty = 'medium'
        else: 
            certainty = 'low'

        n_solvents = 5
        print('ML Predicted Best %i %ss: %s (%s certainty prediction)'%(n_solvents, self.target.capitalize(),', '.join([solvent_names[solvent_order[i]] for i in range(n_solvents)]),certainty))

        return([solvent_names[solvent_order[0]],certainty])

    def make_validation_histogram(self, y_pred_test_all, csv_file, hist_file):

        centroid_dist, filenames = [], []
                
        for y_pred_set in y_pred_test_all:
            centroids = [[np.average([y_pred_set[1][k][i][j] for k in range(len(y_pred_set[1]))]) for j in range(len(y_pred_set[1][0][0]))] for i in range(len(y_pred_set[1][0]))]
            centroid_dist += [np.sqrt(sum([(y_pred_set[1][k][i][j]-centroids[i][j])**2 for j in range(len(centroids[0])) for k in range(len(y_pred_set[1]))])/len(y_pred_set[1])) for i in range(len(centroids))]

            filenames += [self.df['filename'].iloc[idx] for idx in y_pred_set[0]]

        filenames, std = zip(*sorted(zip(filenames, centroid_dist)))

        #for p in range(60,100,5):
        #    print('%2i percentile: %2.3f'%(p,np.percentile(std,p)))

        with open(csv_file,'w') as outfile:
            outfile.write('filename, centroid distance\n')
            for i in range(len(centroid_dist)):
                outfile.write('%s, %s\n'%(filenames[i], centroid_dist[i]))
                
        make_histogram(centroid_dist, 0.01, 'Distances from centroid of the scaled %s models'%self.target, hist_file)


class TT_Model(Regression_Model):

    def __init__(self, target, target_unit = '', *args, **kwargs):

        self.target = target
        self.target_unit = target_unit
        self.feature_names = [target]

        super().__init__(*args, **kwargs)

    def get_final_prediction(self):

        final_prediction = int(np.rint(np.average(self.single_predictions)))
        std = np.std(self.single_predictions)
        
        if std <= 3.0:
            certainty = 'high'
        elif std <= 7.0:
            certainty = 'medium'
        else: 
            certainty = 'low'

        print ('ML Predicted %s: %i %s (%s certainty prediction)'%(self.target.capitalize(), final_prediction, self.target_unit, certainty))

        return([str(final_prediction), certainty])


    def make_validation_histogram(self, y_pred_test_all, csv_file, hist_file):
        
        std, filenames = [], []

        for y_pred_set in y_pred_test_all:
            std += [np.std([y_pred_set[1][j][i] for j in range(len(y_pred_set[1]))]) for i in range(len(y_pred_set[1][0]))]
            filenames += [self.df['filename'].iloc[idx] for idx in y_pred_set[0]]

        filenames, std = zip(*sorted(zip(filenames, std)))

        #for p in range(60,100,5):
        #    print('%2i percentile: %2.3f'%(p,np.percentile(std,p)))

        with open(csv_file,'w') as outfile:
            outfile.write('filename, std\n')
            for i in range(len(std)):
                outfile.write('%s, %s\n'%(filenames[i], std[i]))

        make_histogram(std, 0.5, 'Standard deviation of the %s models'%self.target, hist_file)

