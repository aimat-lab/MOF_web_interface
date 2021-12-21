import pylab
import numpy as np
import pandas as pd

def final_prediction(MOF_random_name, predictions_temperature, predictions_time, predictions_solvent, predictions_additive, startpath = None):

    if startpath is None:
        startpath = os.getcwd()

    #################Temperature##################
    if predictions_temperature is None:
        if os.path.exists('%s/predictions/temperature_prediction.dat'%(startpath)):
            data = pylab.loadtxt('%s/predictions/temperature_prediction.dat'%(startpath))
        else:
            print("ERROR: file not found: predictions/temperature_prediction.dat")
            return(None)
    else:
        data = predictions_temperature
    temperature=sum(data)/len(data)
    temperature_string = "%.2f"%(temperature)

    print ('ML Predcited Temperature :' , round(temperature,1), 'C')


    ####################Time###################
    if predictions_time is None:
        if os.path.exists('%s/predictions/time_prediction.dat'%(startpath)):
            data=pylab.loadtxt('%s/predictions/time_prediction.dat'%(startpath))
        else:
            print("ERROR: file not found: predictions/time_prediction.dat")
            return(None)
    else:
        data = predictions_time

    time=sum(data)/len(data)
    time_string = "%.2f"%(time)
    print ('ML Predcited Time :' , round(time,1), 'Hours')


    ####################Solvent###################
    if predictions_solvent is None:
        if os.path.exists('predictions/solvent_prediction.dat'%(startpath)):
            data=pylab.loadtxt('predictions/solvent_prediction.dat'%(startpath))
        else:
            print("ERROR: file not found: predictions/solvent_prediction.dat")
            return(None)
    else:
        data = predictions_solvent


    df=pd.read_csv("%s/additional_data/local_solvent_full.csv"%(startpath))
    name=df["solvent_name"]

    l=len(data)
    p0=sum(data[:,0])/len(data[:,0])
    p1=sum(data[:,1])/len(data[:,1])
    p2=sum(data[:,2])/len(data[:,2])
    p3=sum(data[:,3])/len(data[:,3])
    p4=sum(data[:,4])/len(data[:,4])


    data_sol=pylab.loadtxt('%s/additional_data/scaled_five_parameter_local_solvent.dat'%(startpath))
    l1=len(data_sol)

    mae=np.zeros(l1)

    for j in range(0,l1):
        mae[j]=(((p0-data_sol[j][0])**2)+((p1-data_sol[j][1])**2)+((p2-data_sol[j][2])**2)+((p3-data_sol[j][3])**2)+((p4-data_sol[j][4])**2))**0.5

    order = np.argsort(mae)

    print ("ML Predicted Five Solvents Are: ",name[order[0]],name[order[1]],name[order[2]],name[order[3]], name[order[4]])
    solvent_string = name[order[0]]

    ####################Additive############################

    if predictions_additive is None:
        if os.path.exists('%s/predictions/additive_prediction.dat'%(startpath)):
            data=pylab.loadtxt('%s/predictions/additive_prediction.dat'%(startpath))
        else:
            print("ERROR: file not found: predictions/additive_prediction.dat")
            return(None)
    else:
        data = predictions_additive


    additive=sum(data)/len(data)

    if additive<0.5:
        print ('ML Predicts the Additive is : A Base' )
        additive_string = "Base"
    elif additive>=0.5 and additive <=1.5:
        print ('ML Predicts the Additive is : Neutral/No Additive' )
        additive_string = "No additive"
    elif additive>1.5:
        print ('ML Predicts the Additive is : An Acid' )
        additive_string = "Acid"


    return([temperature_string, time_string, solvent_string, additive_string])





