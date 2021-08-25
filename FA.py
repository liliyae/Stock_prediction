# Import required libraries
import numpy as np
import pandas as pd
from factor_analyzer import FactorAnalyzer
import matplotlib.pyplot as plt
from sklearn import preprocessing
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo
import seaborn as sns

x= pd.read_csv("fff.csv")

'''
min_max_scaler = preprocessing.MinMaxScaler()
x_minMax = min_max_scaler.fit_transform(x)#归一化

#mean of each feature
n_samples, n_features = x_minMax.shape
mean=np.array([np.mean(x_minMax[:,i]) for i in range(n_features)])  #normalization
x = x_minMax-mean  #去中心化
'''

#chi_square_value,p_value=calculate_bartlett_sphericity(x)
#print(chi_square_value)
#print(p_value)

#kmo_all,kmo_model=calculate_kmo(x)
#print(kmo_model)

# Dropping unnecessary columns
x.drop([],axis=1,inplace=True)
# Dropping missing values rows
x.dropna(inplace=True)
x.head()

#x = np.nan_to_num(x)
#充分性测试(Adequacy Test) kmo>0.6,相关矩阵不是一个identity matrix

# Create factor analysis object and perform factor analysis
fa = FactorAnalyzer(7, rotation="varimax")
print(x)
fa.fit(x)

# Check Eigenvalues
ev, v = fa.get_eigenvalues()

# Create scree plot using matplotlib
plt.scatter(range(1,x.shape[1]+1),ev)
plt.plot(range(1,x.shape[1]+1),ev)
plt.title('Scree Plot')
plt.xlabel('Factors')
plt.ylabel('Eigenvalue')
plt.grid()
plt.show()

print(fa.loadings_)

df_cm = pd.DataFrame(np.abs(fa.loadings_), index=x.columns)
plt.figure(figsize = (14,14))
ax = sns.heatmap(df_cm, annot=True, cmap="BuPu")
# 设置y轴的字体的大小
ax.yaxis.set_tick_params(labelsize=15)
plt.title('Factor Analysis', fontsize='xx-large')
# Set y-axis label
plt.ylabel('Sepal Width', fontsize='xx-large')
plt.savefig('factorAnalysis.png', dpi=500)

fa.transform(x)