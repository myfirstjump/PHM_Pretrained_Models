### Project-LandingGear
### Step1_Data_Preprocessing


### Importing packages
import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

### Define directories
rootDir            = os.getcwd()                            # root directory
trainingHealthyDir = f'{rootDir}\\data\\training\\Healthy'  # training healthy data directory
trainingFaultyDir  = f'{rootDir}\\data\\training\\Faulty'   # training faulty data directory
testingDir         = f'{rootDir}\\data\\testing'            # testing data directory
csvDir             = f'{rootDir}\\csv'                      # final output csv directory

'''
PEP 498 – Literal String Interpolation - f-string
'''

### Read rawdata
def read_rawdata(csv_path: str) -> pd.DataFrame:
    '''function of reading csv rawdata to dataframe'''
    
    flight_num = int(csv_path[-9:-4])                    # get flight number from filename
    plane_num  = str(csv_path.split('\\')[-1][:-10])    # get plane number from filename

    try:
        TO_index = pd.read_csv(csv_path, header=None, usecols=[0], nrows=1).iloc[0, 0]  # get TO data index
        LD_index = pd.read_csv(csv_path, header=None, usecols=[1], nrows=1).iloc[0, 0]  # get LD data index
        # 友達版本
        # Falling back to the 'python' engine because the 'c' engine does not support regex separators (separators > 1 char and different from '\s+' are interpreted as regex); you can avoid this warning by specifying engine='python'.
        # TO_index = pd.read_csv(csv_path, header=None, usecols=[0], nrows=1, engine='python').index[0][0]  # get takeoff data index
        # LD_index = pd.read_csv(csv_path, header=None, usecols=[1], nrows=1, engine='python').index[0][1] # get landing data index
    except Exception as e:
        raise Exception(f'{plane_num}-{flight_num:05d}: error obtaining T/O & L/D data index, {e}') 

    try:
        flight_df = pd.read_csv(csv_path, header=1)     # read rawdata csv
    except Exception as e:
        raise Exception(f'{plane_num}-{flight_num:05d}: error reading rawdata, {e}') 
    
    flight_TO_df = flight_df[:TO_index+1]   # get takeoff data
    flight_LD_df = flight_df[-LD_index:]    # get landing data

    try:
        flight_TO_df = flight_TO_df[flight_TO_df.C > 48]    # select takeoff data (airspeed > 48knots)
        flight_LD_df = flight_LD_df[flight_LD_df.C > 48]    # select landing data (airspeed > 48knots)
    except Exception as e:
        raise Exception(f'{plane_num}-{flight_num:05d}: error selecting data by airspeed, {e}') 

    # reset index, letting index start from 0
    flight_TO_Y_df = flight_TO_df.Y.reset_index(drop=True)
    flight_TO_Z_df = flight_TO_df.Z.reset_index(drop=True)
    flight_LD_Y_df = flight_LD_df.Y.reset_index(drop=True)
    flight_LD_Z_df = flight_LD_df.Z.reset_index(drop=True)

    # set name of the series
    flight_TO_Y_df.name = f'{plane_num}-{flight_num:05d}'
    flight_TO_Z_df.name = f'{plane_num}-{flight_num:05d}'
    flight_LD_Y_df.name = f'{plane_num}-{flight_num:05d}'
    flight_LD_Z_df.name = f'{plane_num}-{flight_num:05d}'
    
    return flight_TO_Y_df, flight_TO_Z_df, flight_LD_Y_df, flight_LD_Z_df


if __name__ == '__main__':
    ## Read healthy rawdata to csv dataset
    # create empty lists for dataframe concat operation
    flight_TO_Y_dfs = []
    flight_TO_Z_dfs = []
    flight_LD_Y_dfs = []
    flight_LD_Z_dfs = []

    # read rawdata by read_rawdata() and append the ouput series to the lists
    for csv in os.listdir(trainingHealthyDir):
        flight_TO_Y_df, flight_TO_Z_df, flight_LD_Y_df, flight_LD_Z_df = read_rawdata(f'{trainingHealthyDir}\\{csv}')
        flight_TO_Y_dfs.append(flight_TO_Y_df)
        flight_TO_Z_dfs.append(flight_TO_Z_df)
        flight_LD_Y_dfs.append(flight_LD_Y_df)
        flight_LD_Z_dfs.append(flight_LD_Z_df)

    # concat all the series in the lists to dateframes
    healthy_TO_Y_Dataset = pd.concat(flight_TO_Y_dfs, axis='columns')
    healthy_TO_Z_Dataset = pd.concat(flight_TO_Z_dfs, axis='columns')
    healthy_LD_Y_Dataset = pd.concat(flight_LD_Y_dfs, axis='columns')
    healthy_LD_Z_Dataset = pd.concat(flight_LD_Z_dfs, axis='columns')

    # save the dataframes to csv files
    healthy_TO_Y_Dataset.to_csv(f'{csvDir}\\healthy_TO_Y_Dataset.csv', index=False)
    healthy_TO_Z_Dataset.to_csv(f'{csvDir}\\healthy_TO_Z_Dataset.csv', index=False)
    healthy_LD_Y_Dataset.to_csv(f'{csvDir}\\healthy_LD_Y_Dataset.csv', index=False)
    healthy_LD_Z_Dataset.to_csv(f'{csvDir}\\healthy_LD_Z_Dataset.csv', index=False)

    ## Read faulty rawdata to csv dataset
    # create empty lists for dataframe concat operation
    flight_TO_Y_dfs = []
    flight_TO_Z_dfs = []
    flight_LD_Y_dfs = []
    flight_LD_Z_dfs = []

    # read rawdata by read_rawdata() and append the ouput series to the lists
    for csv in os.listdir(trainingFaultyDir):
        flight_TO_Y_df, flight_TO_Z_df, flight_LD_Y_df, flight_LD_Z_df = read_rawdata(f'{trainingFaultyDir}\\{csv}')
        flight_TO_Y_dfs.append(flight_TO_Y_df)
        flight_TO_Z_dfs.append(flight_TO_Z_df)
        flight_LD_Y_dfs.append(flight_LD_Y_df)
        flight_LD_Z_dfs.append(flight_LD_Z_df)

    # concat all the series in the lists to dateframes
    faulty_TO_Y_Dataset = pd.concat(flight_TO_Y_dfs, axis='columns')
    faulty_TO_Z_Dataset = pd.concat(flight_TO_Z_dfs, axis='columns')
    faulty_LD_Y_Dataset = pd.concat(flight_LD_Y_dfs, axis='columns')
    faulty_LD_Z_Dataset = pd.concat(flight_LD_Z_dfs, axis='columns')

    # save the dataframes to csv files
    faulty_TO_Y_Dataset.to_csv(f'{csvDir}\\faulty_TO_Y_Dataset.csv', index=False)
    faulty_TO_Z_Dataset.to_csv(f'{csvDir}\\faulty_TO_Z_Dataset.csv', index=False)
    faulty_LD_Y_Dataset.to_csv(f'{csvDir}\\faulty_LD_Y_Dataset.csv', index=False)
    faulty_LD_Z_Dataset.to_csv(f'{csvDir}\\faulty_LD_Z_Dataset.csv', index=False)

    ## Read testing rawdata to csv dataset
    # create empty lists for dataframe concat operation
    flight_TO_Y_dfs = []
    flight_TO_Z_dfs = []
    flight_LD_Y_dfs = []
    flight_LD_Z_dfs = []

    # read rawdata by read_rawdata() and append the ouput series to the lists
    for csv in os.listdir(testingDir):
        flight_TO_Y_df, flight_TO_Z_df, flight_LD_Y_df, flight_LD_Z_df = read_rawdata(f'{testingDir}\\{csv}')
        flight_TO_Y_dfs.append(flight_TO_Y_df)
        flight_TO_Z_dfs.append(flight_TO_Z_df)
        flight_LD_Y_dfs.append(flight_LD_Y_df)
        flight_LD_Z_dfs.append(flight_LD_Z_df)

    # concat all the series in the lists to dateframes
    testing_TO_Y_Dataset = pd.concat(flight_TO_Y_dfs, axis='columns')
    testing_TO_Z_Dataset = pd.concat(flight_TO_Z_dfs, axis='columns')
    testing_LD_Y_Dataset = pd.concat(flight_LD_Y_dfs, axis='columns')
    testing_LD_Z_Dataset = pd.concat(flight_LD_Z_dfs, axis='columns')

    # save the dataframes to csv files
    testing_TO_Y_Dataset.to_csv(f'{csvDir}\\testing_TO_Y_Dataset.csv', index=False)
    testing_TO_Z_Dataset.to_csv(f'{csvDir}\\testing_TO_Z_Dataset.csv', index=False)
    testing_LD_Y_Dataset.to_csv(f'{csvDir}\\testing_LD_Y_Dataset.csv', index=False)
    testing_LD_Z_Dataset.to_csv(f'{csvDir}\\testing_LD_Z_Dataset.csv', index=False)


    ### Plot the healthy and faulty signals
    ## Plot healthy T/O Ny
    plt.figure()
    plt.plot(healthy_TO_Y_Dataset.iloc[:, 5])
    plt.plot(faulty_TO_Y_Dataset.iloc[:, 5],linestyle='-.')
    plt.ylabel('Amplitude')
    plt.xlabel('Data points')
    plt.legend(['healthy','faulty'])
    plt.title('T/O Ny Vibration signals')
    plt.show()

    ## Plot healthy T/O Nz
    plt.figure()
    plt.plot(healthy_TO_Z_Dataset.iloc[:, 5])
    plt.plot(faulty_TO_Z_Dataset.iloc[:, 5],linestyle='-.')
    plt.ylabel('Amplitude')
    plt.xlabel('Data points')
    plt.legend(['healthy','faulty'])
    plt.title('T/O Nz Vibration signals')
    plt.show()

    ## Plot healthy L/D Ny
    plt.figure()
    plt.plot(healthy_LD_Y_Dataset.iloc[:, 5])
    plt.plot(faulty_LD_Y_Dataset.iloc[:, 5],linestyle='-.')
    plt.ylabel('Amplitude')
    plt.xlabel('Data points')
    plt.legend(['healthy','faulty'])
    plt.title('L/D Ny Vibration signals')
    plt.show()

    ## Plot healthy L/D Nz
    plt.figure()
    plt.plot(healthy_LD_Z_Dataset.iloc[:, 5])
    plt.plot(faulty_LD_Z_Dataset.iloc[:, 5],linestyle='-.')
    plt.ylabel('Amplitude')
    plt.xlabel('Data points')
    plt.legend(['healthy','faulty'])
    plt.title('L/D Nz Vibration signals')
    plt.show()
