from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np

#Imputation
from sklearn.linear_model import LinearRegression

#PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def process_bureau_data(df):
    import pandas as pd
    import numpy as np
    
    bureau = df
    
    
    #Processing Numerical Data Features
    bureau_numeric_aggregations = bureau.drop(columns = ['SK_ID_BUREAU']).groupby('SK_ID_CURR', as_index = False).agg(['count','mean','max','min','sum']).T.reset_index().T
    new_columns = []
    for col in bureau_numeric_aggregations.columns:
        col_name = bureau_numeric_aggregations[col].iloc[0]
        agg_name = bureau_numeric_aggregations[col].iloc[1]
        new_columns.append( col_name + '_' + agg_name)


    bureau_numeric_aggregations.columns = new_columns
    bureau_numeric_aggregations = bureau_numeric_aggregations.iloc[2:].reset_index()
    
    #Processing Categorical Data Features
    bureau_dummies = pd.get_dummies(bureau.select_dtypes(exclude=["int64","float64"]))

    bureau_dummies = pd.concat([bureau[['SK_ID_CURR']] ,bureau_dummies], axis = 1)
    bureau_cat_aggs = bureau_dummies.groupby('SK_ID_CURR', as_index = False).agg(['count','sum']).T.reset_index().T

    
    new_columns = []
    for col in bureau_cat_aggs.columns:
        col_name = bureau_cat_aggs[col].iloc[0]
        agg_name = bureau_cat_aggs[col].iloc[1]
        new_columns.append( col_name + '_' + agg_name)

    bureau_cat_aggs.columns = new_columns
    bureau_cat_aggs = bureau_cat_aggs.iloc[2:].reset_index()
    
    
    bureau_df = pd.merge(bureau_cat_aggs, bureau_numeric_aggregations)
    bureau_df = bureau_df.fillna(0)
    
    return bureau_df
    
    
def scale_data(df):
    
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    
    
    scaler = StandardScaler()
    scaler.fit(df)
    
    scaled_data = scaler.transform(df)
    
    return scaled_data
    
    
def perform_PCA(scaled_data , num_of_components):
    
    import pandas as pd
    import numpy as np
    from sklearn.decomposition import PCA
    
    
    pca = PCA(n_components = num_of_components)
    pca.fit(scaled_data)

    bureau_pca = pca.transform(scaled_data)
    pca_b = pd.DataFrame(bureau_pca)
    
    return pca_b
    
def get_top_30_corr(df):
    top_30 = np.abs(df.drop(columns = ['TARGET' ,'AMT_CREDIT']).corr()[['TARGET_X_AMT_CREDIT']]).sort_values('TARGET_X_AMT_CREDIT', ascending = False).head(30)
    df_30 = df[top_30.index]
    
    return df_30



#Hyperparameters for tuning
def model_data(X_train, X_test, y_train, y_test):
    import xgboost as xgb
    from sklearn.ensemble import RandomForestRegressor
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import GridSearchCV




    parameters = {
     'max_depth': [10,None],
     'min_samples_split': [ 5, 10, 15],
     'n_estimators': [100]}

    #Random Forests Regressor
    forest = RandomForestRegressor()

    #GridsearchCV
    reg = GridSearchCV(estimator = forest, 
                       param_grid = parameters,
                       scoring = 'neg_mean_squared_error' ,
                       n_jobs = None ,
                       cv = 2,
                       verbose = 3
                      )

    #Fitting Model
    reg.fit(X_train, y_train)

    return reg
    
    
