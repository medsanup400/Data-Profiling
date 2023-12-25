import pandas as pd
import numpy as np
import os
import tabula
# import matplotlib.pyplot as plt

# from ydata_profiling import ProfileReport

###------------FUNCTIONs-----------_###
#Pushing to medsanup400#


def profile_continuous_variables(df):
    continuous_cols = df.select_dtypes(include=['float', 'int']).columns.tolist()
    profiling_results = {}

    for column in continuous_cols:
        column_stats = {}
        
        column_stats['count'] = df[column].count()
        column_stats['missing_count'] = df[column].isnull().sum()
        column_stats['missing_percentage'] = column_stats['missing_count'] / len(df) * 100
        column_stats['fill_rate'] = column_stats['count']/len(df) * 100
        column_stats['unique_count'] = df[column].nunique()
        column_stats['mean'] = df[column].mean()
        column_stats['median'] = df[column].median()
        column_stats['std'] = df[column].std()
        column_stats['min'] = df[column].min()
        column_stats['25%'] = df[column].quantile(0.25)
        column_stats['50%'] = df[column].quantile(0.50)
        column_stats['75%'] = df[column].quantile(0.75)
        column_stats['max'] = df[column].max()
        column_stats['range'] = column_stats['max'] - column_stats['min']
        column_stats['skewness'] = df[column].skew()
        column_stats['kurtosis'] = df[column].kurtosis()

        # Store the profiling results in the dictionary
        profiling_results[column] = column_stats

    # Return the dictionary containing the profiling results
    return profiling_results


        
def profile_categorical_variables(df):
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    profiling_results = {}

    for column in categorical_cols:
        column_stats = {}
        
        column_stats['count'] = df[column].count()
        column_stats['unique_count'] = df[column].nunique()
        column_stats['distinct_values'] = df[column].unique()
        column_stats['missing_count'] = df[column].isnull().sum()
        # column_stats['missing_percentage'] = column_stats['missing_count'] / len(df) * 100
        column_stats['fill_rate'] = column_stats['count']/len(df) * 100
        
        column_stats['mode'] = df[column].mode().values[0]
        # column_stats['mode_frequency'] = df[column].value_counts().max()
        # column_stats['mode_percentage'] = column_stats['mode_frequency'] / column_stats['count'] * 100
        # column_stats['top_categories'] = df[column].value_counts().head(5).to_dict()

        profiling_results[column] = column_stats

    return profiling_results

def profile_datetime_variables(df):
    datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()

    profiling_results = {}

    for column in datetime_cols:
        column_stats = {}
        
        column_stats['count'] = df[column].count()
        column_stats['missing_count'] = df[column].isnull().sum()
        column_stats['missing_percentage'] = column_stats['missing_count'] / len(df) * 100
        column_stats['earliest_date'] = df[column].min()
        column_stats['latest_date'] = df[column].max()
        column_stats['range'] = column_stats['latest_date'] - column_stats['earliest_date']
        # column_stats['time_unit'] = column_stats['range'].days

        profiling_results[column] = column_stats

    return profiling_results


def profile_boolean_variables(df):
    boolean_cols = df.select_dtypes(include=['bool']).columns.tolist()
    profiling_results = {}
    for column in boolean_cols:
        column_stats = {}

        column_stats['count'] = df[column].count()
        column_stats['missing_count'] = df[column].isnull().sum()
        column_stats['missing_percentage'] = column_stats['missing_count'] / len(df) * 100
        column_stats['true_count'] = df[column].sum()
        column_stats['false_count'] = column_stats['count'] - column_stats['true_count']
        column_stats['true_percentage'] = column_stats['true_count'] / column_stats['count'] * 100
        column_stats['false_percentage'] = 100 - column_stats['true_percentage']

        profiling_results[column] = column_stats

    return profiling_results

    


#--------Read files from the folder----------

folder_path = '/Users/sku/Workspace/IBTS/EPIC/Profiling/Data'  # Specify the path to the folder containing the files

for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)
    
    if file_name.endswith('.csv'):
        df = pd.read_csv(file_path)
        
    elif file_name.endswith('.xlsx') or file_name.endswith('.xls'):
        df = pd.read_excel(file_path)
        
    elif file_name.endswith('.json'):
        df = pd.read_json(file_path, orient='index')
        
        
    elif file_name.endswith('.pdf'):
        pdf_file = 'path_to_pdf_file.pdf'
        tables = tabula.read_pdf(pdf_file, pages='all', multiple_tables=True)
        df = pd.DataFrame()
        for table in tables:
            df = df.append(table, ignore_index=True)
        
    else:
        print(f"Unsupported file format: {file_name}")

    print(file_path)

df = pd.read_csv('/Users/sku/Workspace/IBTS/EPIC/Profiling/Data/DimCustomer.csv')

# change the float display format globally#
pd.set_option('display.float_format', lambda x: '%.2f' % x)

#replace the comma thousands separator with an empty string
df['YearlyIncome'] = df['YearlyIncome'].replace(',', '')

#cast the columns to the correct datatypes
df = df.astype({'BirthDate':'datetime64','DateFirstPurchase':'datetime64', 'YearlyIncome':'float64'})

##----Continous variables profiling----
cont_profile = profile_continuous_variables(df)
dfj_cont = pd.DataFrame.from_dict(cont_profile, orient='index')
print(f"\nContinous Variables profiling:\n {dfj_cont}\n")



##------Categorical variables profiling---------
catg_profile = profile_categorical_variables(df)
dfj_catg = pd.DataFrame.from_dict(catg_profile, orient='index')
print(f"\nCategorical Variables profiling:\n {dfj_catg}\n")


##-------Datetime variables profiling---------------

dtm_profile = profile_datetime_variables(df)
dfj_dtm = pd.DataFrame.from_dict(dtm_profile, orient='index')
print(f"\nDatetime Variables profiling:\n {dfj_dtm}\n")





####----------ANALYSIS-----------###



# Print the data types of columns
# print(df.dtypes)

#change the float display format globally#
# pd.set_option('display.float_format', lambda x: '%.2f' % x)

# #replace the comma thousands separator with an empty string
# df['YearlyIncome'] = df['YearlyIncome'].replace(',', '')

# #cast the columns to the correct datatypes
# df = df.astype({'BirthDate':'datetime64','DateFirstPurchase':'datetime64', 'YearlyIncome':'float64'})

# Print the data types of columns
# print(df.dtypes)

# print(df.describe())



#---------------Histogram-------------#
#square root choice
# bins = int(np.ceil(np.sqrt(len(num_cols['YearlyIncome']))))

# fig, ax = plt.subplots(1, 1, figsize=(14,8))
# ax.hist(num_cols['YearlyIncome'])#, bins)


# # ax.ticklabel_format(useOffset=False, style='plain')
# ax.set_xlabel('YearlyIncome')
# ax.set_ylabel('Count')
# ax.set_title(r'Histogram of YearlyIncome in USD')
# plt.show()

