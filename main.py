import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# load data
def load_data(file_path, year):
    """
    Load data from a CSV file into a pandas DataFrame.
    
    Parameters:
    file_path (str): The path to the CSV file.
    
    Returns:
    pd.DataFrame: DataFrame containing the loaded data.
    """
    try:
        data = pd.read_csv(file_path)
        data['Year'] = year
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

df_2015 = load_data('2015_cleaned.csv', 2015)
df_2016 = load_data('2016_cleaned.csv', 2016)
df_2017 = load_data('2017_cleaned.csv', 2017)
df_2018 = load_data('2018_cleaned.csv', 2018)
df_2019 = load_data('2019_cleaned.csv', 2019)

# correlation_matrix_2015 = df_2015.corr(numeric_only=True)
# print("\nCorrelation Matrix for 2015:")
# print(correlation_matrix_2015)
# sns.heatmap(correlation_matrix_2015, annot=True, cmap='coolwarm')
# plt.show()


# or drop Overall Rank
cols_to_use = [
    'Economy (GDP per Capita)',
    'Family',
    'Freedom',
    'Generosity',
    'Health (Life Expectancy)',
    'Trust (Government Corruption)',
    'Year'
]

target_column = 'Happiness Score'

df = pd.concat([df_2015, df_2016, df_2017, df_2018, df_2019], ignore_index=True)
df = df.sort_values(by=['Country', 'Year'])

# drop 'Overall Rank' column if it exists
if 'Overall Rank' in df.columns:
    df = df.drop('Overall Rank', axis=1)

# create lag features for each numeric column
    for col in cols_to_use[:-1]:  # exclude 'Year'
        df[f'{col}_lag1'] = df.groupby('Country')[col].shift(1)
        df[f'{col}_lag2'] = df.groupby('Country')[col].shift(2)

    # drop rows with NaN values from lag features
    df = df.dropna()

    # update cols_to_use to include lag features
    cols_to_use = cols_to_use + [f'{col}_lag1' for col in cols_to_use[:-1]] + [f'{col}_lag2' for col in cols_to_use[:-1]]

# define X and y
X = df[cols_to_use]
y = df[target_column]


# save DataFrame to CSV
df.to_csv('happiness_combined.csv', index=False)

# separate 2019 data
X_train = X[X['Year'] < 2019]
y_train = y[X['Year'] < 2019]

X_test = X[X['Year'] == 2019]
y_test = y[X['Year'] == 2019]

# train the model
xgb_model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
xgb_model.fit(X_train, y_train)

# predictions
y_pred = xgb_model.predict(X_test)

# evaluation
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.4f}")
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")

# train model on all data
X_full = X
y_full = y
xgb_model.fit(X_full, y_full)

# create 2020 predictions dataframe
X_2020 = X[X['Year'] == 2019].copy()  # use 2019 data as baseline
X_2020['Year'] = 2020  # update year to 2020

# make predictions
predictions_2020 = xgb_model.predict(X_2020)

# create results dataframe
results_2020 = pd.DataFrame({
    'Country': df[X['Year'] == 2019]['Country'].values,
    'Predicted_Happiness_2020': predictions_2020
})

# sort predictions by happiness score in descending order
results_2020 = results_2020.sort_values('Predicted_Happiness_2020', ascending=False)

# save predictions to CSV 
results_2020.to_csv('2020_prediction.csv', index=False)