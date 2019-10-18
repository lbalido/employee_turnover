import pandas as pd
import numpy as np
from datetime import datetime, timedelta 
import os

from sklearn.metrics import log_loss, make_scorer, confusion_matrix, classification_report


def weekends(df):
    #Weekend = 1, Weekday = 0
    df['Official Clock In Date']= pd.to_datetime(df['Official Clock In Date']) 
    df['dayofweek'] = df['Official Clock In Date'].dt.dayofweek
    df['Weekend'] = np.where(df['dayofweek'] <5 , 'Weekday', 'Weekend')
    
    return df.drop(columns='dayofweek')


def caregivers_system2(col_order):
    
    '''
    Returns caregivers from system 2 in format of system 1
    '''
    
    ## read both active and inactive
    active_cgs = pd.read_excel('data/active_cgs.xlsx')
    inactive_cgs = pd.read_excel('data/inactive_cgs.xlsx')
    inactive_cgs2 = pd.read_excel('data/inactive_cgs2.xlsx')
    
    ## add Inactive column
    active_cgs['Inactive'] = False
    inactive_cgs['Inactive'] = True
    inactive_cgs2['Inactive'] = True
    
    
    ## create new merged dataframe
    cgs2 = pd.DataFrame()
    cgs2 = cgs2.append(active_cgs)
    cgs2 = cgs2.append(inactive_cgs)
    cgs2 = cgs2.append(inactive_cgs2)
    
    
    ## update columns to match previous system
    cgs2 = cgs2.drop(columns = ['Unnamed: 0', 'City', 'Postal Code'])

    cgs2 = cgs2.rename(columns={'ID': 'Payroll ID', 'Date of Birth' : 'Date Of Birth', 'Sex' : 'Gender',
                           'First Service Date' : 'First Care Log Date', 
                            'Last Service Date' : 'Last Care Log Date', 'Separation Date' : 'Termination Date',
                           'Office Area' : 'Location', 'Skill Category' : 'Caregiver Tags'  })
    
    
    cgs2 = cgs2[cgs2['Date Of Birth'] != None]
    
    cgs2 = cgs2[col_order]
    
    
    
    return cgs2



def clean_cgs(df):
    
    ## change CG tags to CNA True/False. True = 1, False = 0
    df['CNA'] = np.where(df['Caregiver Tags'].str.contains('CNA'), 1, 0)
    df = df.drop(columns='Caregiver Tags')
    
    
    ## change inactive tag to 0/1. True = 1, False = 0
    df['Inactive'] = np.where(df['Inactive'] == True, 1, 0)
    
    
    # add Age, drop DOB
    now = pd.Timestamp('now')
    df['Date Of Birth'] = pd.to_datetime(df['Date Of Birth'], format='%y%m%d')    
    df['Date Of Birth'] = df['Date Of Birth'].where(df['Date Of Birth'] < now, df['Date Of Birth'] -  np.timedelta64(100, 'Y'))
    df['Age'] = (now - df['Date Of Birth']).astype('<m8[Y]') 
    df['Age'] = df['Age'].fillna(df['Age'].mean())
    
    df = df.drop(columns='Date Of Birth')
    
    
    ## change gender to 0/1. Female = 1, Male = 0
    # NaN default to female -- industry is 90% female.
    df['Gender'] = np.where(df['Gender'] == 'Male', 0, 1)
    
    
    ## drop payroll id and location nans
    df = df[pd.notnull(df['Payroll ID'])]
    df = df[pd.notnull(df['Location'])]


    ## change state tab to 0/1. LA = 1, MS = 0
    df['State'] = np.where(df['State'] == 'LA', 1, 0)
    
    return df



def df_concatenate(path):
    documents = {}
    file_names = [f for f in os.listdir(path) if os.path.isfile((path) + '/' + f)]

    df = pd.DataFrame()

    for f in file_names:
        data = pd.read_excel(path+f)
        df = df.append(data)


    return df


def clean_carelogs(carelogs):

    carelogs = carelogs[carelogs['Status'] == 'Complete']
    carelogs = carelogs.rename(columns={'Caregiver ID': 'Payroll ID'})
    carelogs = carelogs[carelogs['Payroll ID'] != 'chrissy@afirstnamebasis.com']
    carelogs['Payroll ID'] = pd.to_numeric(carelogs['Payroll ID'])
    carelogs['Bill Rate Name'] = np.where(carelogs['Bill Rate Name'].isnull(), carelogs['Pay Rate Name'], carelogs['Bill Rate Name'])
    carelogs['Bill Rate Name'] = np.where(carelogs['Bill Rate Name'].isnull(), 'Private', carelogs['Bill Rate Name'])

    carelogs = carelogs.drop(columns= ['Status', 'Pay Rate Name', 'Pay OT Hours', 'Pay Reg Hours',
                                            'Official Clock In Time', 'Client Tags','Pay Total'])

    return weekends(carelogs)



def clean_payroll(df):

    ## rename columns
    df = df.rename(columns = {'textbox46' : 'Payroll ID', 'textbox121' : 'Official Clock In Date',  
    'textbox30' : 'Bill Rate Name', 'textbox21' : 'Pay Rate Amount', 'textbox26' : 'Pay Total Hours'})
    
    ## drop unneccessary columns
    drops = [i for i in df.columns if 'textbox' in i]
    df = df.drop(columns = drops)
    df = df.drop(columns = ['Unnamed: 0', 'Textbox70'])

    ## Update ID# column
    new = df['Payroll ID'].str.split(":", n = 1, expand = True) 
    df['Payroll ID']= new[1].astype(float)
    
    ## Remove unwanted info
    df = df[df['Bill Rate Name'] != 'WeeklyOT']
    df = df[df['Bill Rate Name'] != 'Weekly OT Adj.']
    df = df[df['Bill Rate Name'] != 'Mileage']
    df = df[df['Bill Rate Name'] != 'Miscellaneous Addition']
    df = df[df['Bill Rate Name'] != 'Incentive']
    df = df[df['Bill Rate Name'] != 'First Aid']
    df = df[df['Bill Rate Name'] != 'Referral Bonus']
    df = df[df['Bill Rate Name'] != 'TB']
    df = df[df['Bill Rate Name'] != 'Miscellaneous']
    df = df[df['Bill Rate Name'] != 'Home Visit']
    
    ## Update pay rate amount type
    df['Pay Rate Amount'] = df['Pay Rate Amount'].replace({'\$': ''}, regex=True).astype(float)


    ## Update Live-In Rates
    df['Pay Rate Amount'] = df.apply(lambda x: 8 if x['Bill Rate Name'] == 'Live-In' else x['Pay Rate Amount'], axis=1)


    return weekends(df)




def first_shift(df, n_days):
    
    ## {1: first shift was longer than n_days, 0: first shift was within n_days}
    df['first_shift_within_{}_days'.format(n_days)] = np.where(df['First Care Log Date'] - df['Hire Date'] <= timedelta(days=n_days), 1, 0)
    return df



def dummies(df, col_name):
    return pd.concat([df.drop(col_name, axis=1), pd.get_dummies(df[col_name])], axis=1)



def nulls(df, col):
    return df[df[col].isnull()].count()['Payroll ID']



