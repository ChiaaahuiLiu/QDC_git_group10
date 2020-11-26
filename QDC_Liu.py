# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 17:17:06 2020

@author: Chiahui Liu
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import sklearn.ensemble as ensemble 
from sklearn.model_selection import GridSearchCV
from typing import Union
from sklearn import metrics
from sklearn.metrics import auc,roc_curve
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from sklearn.inspection import permutation_importance



#data loading
df_full_target=pd.read_csv('QTEM_TARGET.csv')
print(df_full_target.head(3))
print('Shape of target data :',df_full_target.shape)
df_full_data =pd.read_csv('QTEM_DATABASE.csv')
print('Shape of full data :',df_full_data.shape)

#data preprocessing_combine 2 data sets_borrowed
df_full_data['IS_TARGET'] = np.where(df_full_data.index.isin(df_full_target.index), 1, 0)


def get_frequencies(data: pd.DataFrame, n_categories: int = None,
                    bins: int = None, dropna: bool = False
                    ):
    for (name, val) in data.iteritems():
        print('')
        print(name)
        vc = val.value_counts(ascending=False,
                              bins=bins,
                              dropna=dropna
                              )
        if n_categories is not None:
            if not isinstance(n_categories, int) or n_categories <= 0:
                raise TypeError(
                    'n_categories should be a strictly positive integer')
            if n_categories < len(vc):
                freq_others = vc.iloc[n_categories - 1:].sum()
                vc = vc.iloc[:n_categories - 1] \
                    .append(pd.Series({'others': freq_others}))
        print(pd.DataFrame({'absolute': vc,
                            'relative': vc / len(val) * 100,
                            },
                           index=vc.index
                           ).T)
print('Shape of full data_new_1 :', df_full_data.shape)

#data preprocessing_data cleaning_borrowed

def summarize_na(df: pd.DataFrame) -> pd.DataFrame:
    nan_count = df.isna().sum()
    nan_pct = nan_count / len(df) * 100
    return pd.DataFrame({'nan_count': nan_count,
                         'nan_pct': nan_pct
                         }
                        )[nan_pct >50]
    
df_full_data = df_full_data.drop(summarize_na(df_full_data).index, axis=1)
print('Shape of full data_new_2:', df_full_data.shape)

#data parcialing_borrowed
df_cat = df_full_data.select_dtypes(include="object").copy()
df_num = df_full_data.select_dtypes(exclude="object").copy()

def encode_onehot(series: pd.Series, drop_last: bool = True
                  ) -> Union[pd.Series, pd.DataFrame]:
     values = series.unique()
     if drop_last:
         values = values[:-1]
     return pd.concat(((series == val).rename(val)
                      for val in values
                       ),
                      axis=1
                      ).squeeze()

df_num['IS_FEMALE'] = encode_onehot(df_full_data['CD_GENDER'])
pd.set_option("display.max_rows", None)
df_num = df_num.dropna()
X = df_num.drop('IS_TARGET', axis=1)
y = df_num.IS_TARGET


# random forest model
 
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                    test_size=0.4, random_state=12345)
param_grid = {
    'criterion':['entropy','gini'],
    'max_depth':[5, 6, 7, 8],    
    'n_estimators':[11,13,15],  
    'max_features':[0.3,0.4,0.5],
    'min_samples_split':[4,8,12,16]  
}

rfc = ensemble.RandomForestClassifier()
rfc_cv = GridSearchCV(estimator=rfc, param_grid=param_grid,
                      scoring='roc_auc', cv=5)
rfc_cv.fit(X_train, y_train)
test_est = rfc_cv.predict(X_test)

print("Accuracy on test data: {:.2f}".format(rfc.score(X_test, y_test)))
print(metrics.classification_report(test_est, y_test))
fpr_test, tpr_test, th_test = metrics.roc_curve(test_est, y_test)
print('AUC = %.4f'%metrics.auc(fpr_test, tpr_test))

#plot roc

y_label = ([1, 1, 1, 2, 2, 2]) 
y_pre = ([0.3, 0.5, 0.9, 0.8, 0.4, 0.6])
fpr_test, tpr_test, thersholds = roc_curve(y_label, y_pre, pos_label=2)
 
for i, value in enumerate(thersholds):
    print("%f %f %f" % (fpr_test[i], tpr_test[i], value))
 
roc_auc = auc(fpr_test, tpr_test)
 
plt.plot(fpr_test, tpr_test, 'k--', label='ROC (area = {0:.2f})'.format(roc_auc), lw=2)
 
plt.xlim([-0.05, 1.05])  
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate') 
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

#Plot Random Forest Feature Importance 

result = permutation_importance(rfc, X_train, y_train, n_repeats=10,
                                random_state=42)
perm_sorted_idx = result.importances_mean.argsort()

tree_importance_sorted_idx = np.argsort(rfc.feature_importances_)
tree_indices = np.arange(0, len(rfc.feature_importances_)) + 0.5

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
ax1.barh(tree_indices,
         rfc.feature_importances_[tree_importance_sorted_idx], height=0.7)
ax1.set_yticklabels(df_num.feature_names[tree_importance_sorted_idx])
ax1.set_yticks(tree_indices)
ax1.set_ylim((0, len(rfc.feature_importances_)))
ax2.boxplot(result.importances[perm_sorted_idx].T, vert=False,
            labels=df_num.feature_names[perm_sorted_idx])
fig.tight_layout()
plt.show()