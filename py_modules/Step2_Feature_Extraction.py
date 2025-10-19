### Project-LandingGear
### Step2_Feature_Extraction


### Importing packages
import os

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

### Define directories
rootDir            = os.getcwd()                            # root directory
trainingHealthyDir = f'{rootDir}\\data\\training\\Healthy'  # training healthy data directory
trainingFaultyDir  = f'{rootDir}\\data\\training\\Faulty'   # training faulty data directory
testingDir         = f'{rootDir}\\data\\testing'            # testing data directory
csvDir             = f'{rootDir}\\csv'                      # csv dataset directory
featuresDir        = f'{rootDir}\\myfeature'                # final output features directory


### Define features name
feature_name=['1-TO-Y-rms','2-TO-Y-mean','3-TO-Y-std','4-TO-Y-peak2peak','5-TO-Y-kurtosis','6-TO-Y-skewness','7-TO-Y-crest_indicator','8-TO-Y-clearance_indicator','9-TO-Y-shape_indicator','10-TO-Y-impulse_indicator',
              '11-TO-Z-rms','12-TO-Z-mean','13-TO-Z-std','14-TO-Z-peak2peak','15-TO-Z-kurtosis','16-TO-Z-skewness','17-TO-Z-crest_indicator','18-TO-Z-clearance_indicator','19-TO-Z-shape_indicator','20-TO-Z-impulse_indicator',
              '21-LD-Y-rms','22-LD-Y-mean','23-LD-Y-std','24-LD-Y-peak2peak','25-LD-Y-kurtosis','26-LD-Y-skewness','27-LD-Y-crest_indicator','28-LD-Y-clearance_indicator','29-LD-Y-shape_indicator','30-LD-Y-impulse_indicator',
              '31-LD-Z-rms','32-LD-Z-mean','33-LD-Z-std','34-LD-Z-peak2peak','35-LD-Z-kurtosis','36-LD-Z-skewness','37-LD-Z-crest_indicator','38-LD-Z-clearance_indicator','39-LD-Z-shape_indicator','40-LD-Z-impulse_indicator']


### Extract features
def extract_features(dataset_path: str) -> pd.DataFrame:
    '''function of extracting features from csv dataset'''

    dataset = pd.read_csv(dataset_path) # read csv dataset to dataframe
    numOfData = dataset.shape[1]        # number of data
    numOfFeature = 10                   # number of feature

    features = np.zeros((numOfData, numOfFeature))  # create a '(number of data) x (number of feature)' ndarray of zeros
    for m in range(numOfData):
        data = dataset.iloc[:, m].dropna().values   # dropping nan

        features[m, 0] = np.sqrt(np.mean(data**2))                              # rms
        features[m, 1] = np.mean(data)                                          # mean
        features[m, 2] = np.std(data)                                           # std
        features[m, 3] = data.max() - data.min()                                # peak2peak
        features[m, 4] = kurtosis(data, fisher=False)                           # kurtosis
        features[m, 5] = skew(data)                                             # skewness
        features[m, 6] = abs(data.max()) / np.sqrt(np.mean(data ** 2))          # crest indicator
        features[m, 7] = abs(data.max()) / np.mean(np.sqrt(abs(data))) ** 2     # clearance indicator
        features[m, 8] = np.sqrt(np.mean(data ** 2)) / np.mean(abs(data))       # shape indicator
        features[m, 9] = abs(data.max()) / np.mean(abs(data))                   # impulse indicator

    return pd.DataFrame(features)

if __name__ == '__main__':
    ## Extract features from healthy dataset
    healthy_dataset_list = ['healthy_TO_Y_Dataset.csv', 'healthy_TO_Z_Dataset.csv', 'healthy_LD_Y_Dataset.csv', 'healthy_LD_Z_Dataset.csv']
    tmp_dfs = []
    for dataset in healthy_dataset_list:
        tmp_features = extract_features(f'{csvDir}\\{dataset}')
        tmp_dfs.append(tmp_features)

    healthyAllFeatures = pd.concat(tmp_dfs, axis='columns')
    healthyAllFeatures.columns = feature_name
    healthyAllFeatures.to_csv(f'{featuresDir}\\healthyAllFeatures.csv', index=False)

    ## Extract features from faulty dataset
    faulty_dataset_list = ['faulty_TO_Y_Dataset.csv', 'faulty_TO_Z_Dataset.csv', 'faulty_LD_Y_Dataset.csv', 'faulty_LD_Z_Dataset.csv']
    tmp_dfs = []
    for dataset in faulty_dataset_list:
        tmp_features = extract_features(f'{csvDir}\\{dataset}')
        tmp_dfs.append(tmp_features)

    faultyAllFeatures = pd.concat(tmp_dfs, axis='columns')
    faultyAllFeatures.columns = feature_name
    faultyAllFeatures.to_csv(f'{featuresDir}\\faultyAllFeatures.csv', index=False)

    ## Extract features from testing dataset
    testing_dataset_list = ['testing_TO_Y_Dataset.csv', 'testing_TO_Z_Dataset.csv', 'testing_LD_Y_Dataset.csv', 'testing_LD_Z_Dataset.csv']
    tmp_dfs = []
    for dataset in testing_dataset_list:
        tmp_features = extract_features(f'{csvDir}\\{dataset}')
        tmp_dfs.append(tmp_features)

    testingAllFeatures = pd.concat(tmp_dfs, axis='columns')
    testingAllFeatures.columns = feature_name
    testingAllFeatures.to_csv(f'{featuresDir}\\testingAllFeatures.csv', index=False)
