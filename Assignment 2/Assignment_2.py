
## importing required libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn import model_selection
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import optuna.integration.lightgbm as 

## reading test and train datasets
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

## function for filling missing values with mode values of respective columns
def handle_missing_values(train, test, features): 
    for feature in features:
        train[feature].fillna(train[feature].mode()[0], inplace=True)
        test[feature].fillna(test[feature].mode()[0], inplace=True)
                
    return train,test

## filling values with lion as 1 as others as 0, same change reflected in test dataset
def filter_batteries(bat):
    bat = str(bat)
    if 'li' in bat:
        return 1
    else:
        return 0


def preprocessing():

	## Combining vague values of Smartscreen column into smaller groups of values
	conversion_dict = {
    'off': 'Off', '&#x02;': '2', '&#x01;': '1', 'on': 'On', 'requireadmin': 'RequireAdmin', 'OFF': 'Off', 
    'Promt': 'Prompt', 'requireAdmin': 'RequireAdmin', 'prompt': 'Prompt', 'warn': 'Warn', 
    '00000000': '0', '&#x03;': '3', np.nan: 'NoExist'
	}

	train.replace({'SmartScreen': conversion_dict}, inplace=True)
	test.replace({'SmartScreen': conversion_dict}, inplace=True)

	## Merging different types of unknows of Disk Types into single 'Unknown' value
	conversion_dict = {
    'UNKNOWN': 'unknown', 'Unspecified': 'unknown'
	}
	train.replace({'Census_PrimaryDiskTypeName': conversion_dict}, inplace=True)
	test.replace({'Census_PrimaryDiskTypeName': conversion_dict}, inplace=True)

	## Merging different types of values of PowerPlatfromRole column into similar smaller groups of values 
	conversion_dict = {
    	'AppliancePC' : 'AppliancePC', 'Desktop' : 'Desktop', 'EnterpriseServer' : 'EnterpriseServer', 
    	'Mobile' : 'Mobile', 'SOHOServer' : 'SOHOServer','Slate' : 'Slate', 'Workstation' : 'Workstation', 
    	'PerformanceServer' : 'PerformanceServer', 'UNKNOWN': 'UNKNOWN', 'Unspecified': 'UNKNOWN'
	}
	train.replace({'Census_PowerPlatformRoleName': conversion_dict}, inplace=True)
	test.replace({'Census_PowerPlatformRoleName': conversion_dict}, inplace=True)

	## Merging different types of values of MDC2FormFactor column into similar smaller groups of values
	conversion_dict = {
    	'AllInOne' : 'AllInOne', 'Convertible' : 'Convertible', 'Desktop' :  'Desktop', 'Detachable' : 'Detachable', 
    	'LargeServer' : 'Server','LargeTablet' : 'Tablet', 'MediumServer' : 'Server', 'Notebook' : 'Notebook', 
    	'PCOther' : 'PCOther', 'SmallServer' : 'Server','SmallTablet' : 'Tablet'
	}
	train.replace({'Census_MDC2FormFactor': conversion_dict}, inplace=True)
	test.replace({'Census_MDC2FormFactor': conversion_dict}, inplace=True)


def additional_features(train, test):
    
    # Disk Space Remaining
    train['disk_remain'] = train['Census_PrimaryDiskTotalCapacity'] - train['Census_SystemVolumeTotalCapacity']
    test['disk_remain'] = test['Census_PrimaryDiskTotalCapacity'] - test['Census_SystemVolumeTotalCapacity']
    train['disk_remain'] = train['disk_remain'].astype('float32')
    test['disk_remain'] = test['disk_remain'].astype('float32')

    # Ram-to-CPU ratio
    train['ram_cpu_ratio'] = train['Census_TotalPhysicalRAM'] / train['Census_ProcessorCoreCount']
    test['ram_cpu_ratio'] = test['Census_TotalPhysicalRAM'] / test['Census_ProcessorCoreCount']

    # Pixel Per Inch PPI sqrt(horizonal**2 + vertical**2) / diagonal
    train['ppi'] = np.sqrt(train['Census_InternalPrimaryDisplayResolutionHorizontal']**2 + train['Census_InternalPrimaryDisplayResolutionVertical']**2) / train['Census_InternalPrimaryDiagonalDisplaySizeInInches']
    test['ppi'] = np.sqrt(test['Census_InternalPrimaryDisplayResolutionHorizontal']**2 + test['Census_InternalPrimaryDisplayResolutionVertical']**2) / test['Census_InternalPrimaryDiagonalDisplaySizeInInches']

    # Screen aspect ratio = Horizonal / Vertical
    train['aspect_ratio'] = train['Census_InternalPrimaryDisplayResolutionHorizontal'] / train['Census_InternalPrimaryDisplayResolutionVertical']
    test['aspect_ratio'] = test['Census_InternalPrimaryDisplayResolutionHorizontal'] / test['Census_InternalPrimaryDisplayResolutionVertical']

    # Pixel count = Horizonal * Vertical
    train['pixel_count'] = train['Census_InternalPrimaryDisplayResolutionHorizontal'] * train['Census_InternalPrimaryDisplayResolutionVertical']
    test['pixel_count'] = test['Census_InternalPrimaryDisplayResolutionHorizontal'] * test['Census_InternalPrimaryDisplayResolutionVertical']

    ## deleting redundant columns which we already used for making new features
	train.drop('Census_PrimaryDiskTotalCapacity', axis=1, inplace=True)
	test.drop('Census_PrimaryDiskTotalCapacity', axis=1, inplace=True)
	train.drop('Census_ProcessorClass',inplace=True, axis=1)
	test.drop('Census_ProcessorClass', inplace=True, axis=1)
    
    return train, test	


def frequency_encoding(variable):
    t = pd.concat([train[variable], test[variable]]).value_counts().reset_index()
    t = t.reset_index()
    t.loc[t[variable] == 1, 'level_0'] = np.nan
    t.set_index('index', inplace=True)
    max_label = t['level_0'].max() + 1
    t.fillna(max_label, inplace=True)
    return t.to_dict()['level_0']

def label_encoding(train, test, label_encoded):
	for feature in label_encoded:
		train[feature] = LabelEncoder().fit_transform(train[feature])
		test[feature] = LabelEncoder().fit_transform(test[feature])
	return train, test


def main():

	preprocessing()

	## filling missing values with mode value for following columns
	missing_value_features = ["DefaultBrowsersIdentifier","Census_IsFlightingInternal", "Census_ThresholdOptIn", "Census_IsWIMBootEnabled", "OrganizationIdentifier",
                "SMode", "Wdft_IsGamer", "Wdft_RegionIdentifier", "Census_FirmwareManufacturerIdentifier", "Census_FirmwareVersionIdentifier", 
                "Census_OEMModelIdentifier", "Census_OEMNameIdentifier", "Firewall", "Census_TotalPhysicalRAM", "Census_IsAlwaysOnAlwaysConnectedCapable",
                "Census_OSInstallLanguageIdentifier", "IeVerIdentifier", "Census_SystemVolumeTotalCapacity", "Census_PrimaryDiskTotalCapacity",
                "Census_InternalPrimaryDiagonalDisplaySizeInInches", "Census_InternalPrimaryDisplayResolutionHorizontal", 
                "Census_InternalPrimaryDisplayResolutionVertical", "AVProductsEnabled", "AVProductsInstalled", "AVProductStatesIdentifier", "IsProtected", 
                "Census_ProcessorModelIdentifier", "Census_ProcessorCoreCount", "Census_ProcessorManufacturerIdentifier", "RtpStateBitfield", 
                "Census_IsVirtualDevice", "UacLuaenable", "GeoNameIdentifier",'SmartScreen', 'Census_PrimaryDiskTypeName', 'Census_ChassisTypeName', 
                'Census_PowerPlatformRoleName', 'OsBuildLab']

	train, test = handle_missing_values(train, test, missing_value_features)

	## filtering values of Battery_Type column 
	train["Census_InternalBatteryType"] = train["Census_InternalBatteryType"].apply(filter_batteries)
	test["Census_InternalBatteryType"] = test["Census_InternalBatteryType"].apply(filter_batteries)     

	preprocessing()  ## processing some columns by merging similar values into smaller groups / filtering etc.

	## These columns are decided to be dropped on the basis of analysis of correlation, skewness, unbalancing, percent of 
	## missing values present in them
	columns_to_drop = ["IsBeta", "Census_IsWIMBootEnabled", "Census_IsFlightingInternal", "Census_ThresholdOptIn", 
                   "Census_IsPortableOperatingSystem", "SMode", "Census_DeviceFamily", "UacLuaenable", 
                   "Census_IsVirtualDevice", "ProductName", "PuaMode", ]

	## dropping columns from both train and test datasets to maintain equality between the two datasets 
    train.drop(columns_to_drop, axis=1, inplace=True)
	test.drop(columns_to_drop, axis=1, inplace=True)

	## Feature Engineering
	## adding new features to both test and train datasets.
	train, test = additional_features(train, test)

	## Dropping these columns form train and test datasets because of high-correlation, resulting EDA analysis etc 
	additional_columns_to_drop = ["Census_ChassisTypeName", "Census_OSVersion", "IsSxsPassiveMode", "Census_ProcessorManufacturerIdentifier", "Census_InternalPrimaryDisplayResolutionVertical", 
										"Census_OSInstallLanguageIdentifier", "MachineIdentifier", "CityIdentifier", "AutoSampleOptIn", "Census_InternalBatteryType", 
										"Census_InternalBatteryNumberOfCharges", "Census_OSArchitecture", "Census_OSSkuName", "Census_IsFlightsDisabled"]

	train.drop(additional_columns_to_drop, axis=1, inplace=True)
	test.drop(additional_columns_to_drop, axis=1, inplace=True)

	## For converting categorical values into numerical, we will be using Frequency encoding, which gives numnerical value according to frequency(no of times) the particular 
	## value occurs in the column. some other features we will also be using Label Encoding.
	frequency_encoded = ['OsPlatformSubRelease', 'OsBuildLab', 'Processor', 'SkuEdition', 'SmartScreen', 'Platform']

	for variable in frequency_encoded:
    	freq_enc_dict = frequency_encoding(variable)
    	train[variable] = train[variable].map(lambda x: freq_enc_dict.get(x, np.nan))
    	test[variable] = test[variable].map(lambda x: freq_enc_dict.get(x, np.nan))

    label_encoded = ['Census_MDC2FormFactor', 'Census_PrimaryDiskTypeName', 'Census_PowerPlatformRoleName', 'Census_OSBranch', 'Census_OSEdition', 
                   'Census_OSInstallTypeName', 'Census_OSWUAutoUpdateOptionsName', 'Census_GenuineStateName', 'Census_ActivationChannel', 
                   'Census_FlightRing', 'EngineVersion', 'AppVersion', 'AvSigVersion', 'OsVer']

    train, test = label_encoding(train, test, label_encoded)

    ## Seperating target feature "HasDetections" from training dataset
    target = train_db['HasDetections']
	train_db.drop('HasDetections', axis=1, inplace=True)

	## splitting train dataaset for validation
	x_train, x_val, y_train, y_val = train_test_split(train, target, test_size = 0.15, random_state = 1)

	# Using Optuna library for tuning hyperparameters
	dtrain = lgbm.Dataset(x_train, label=y_train)
	dval = lgbm.Dataset(x_val, label=y_val)

	params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "boosting_type": "gbdt",
    }

    ## finding best possible combination of hype-rprameters
	model = lgbm.train(params, dtrain, valid_sets=[dtrain, dval], verbose_eval=100, early_stopping_rounds = 5)

	prediction = np.rint(model.predict(x_val, num_iteration = model.best_iteration))
	accuracy = accuracy_score(y_val, prediction)

	## best hyper-parameters

	best_params = model.params
	print("  Params: ")
	for key, value in best_params.items():
    	print("    {}: {}".format(key, value))

    ## training model using best obtained (tuned) hyper-parameters
	model = lgbm.train(best_params, dtrain, 10000, valid_sets=[dtrain, dval], early_stopping_rounds=500, verbose_eval=100)

	## predicting with best iteration score obtained above
	result = model.predict(test_db, num_iteration = model.best_iteration)

	## saving prediction results to csv file
	df = pd.DataFrame(result)
	res = pd.concat([machine_id, df], axis = 1)
	res.set_index('MachineIdentifier',inplace=True)
	res.columns = {'HasDetections'}
	res.to_csv('submission_kaggle.csv')

main()










