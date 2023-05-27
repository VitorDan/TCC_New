from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import train_test_split
from numpy import absolute, mean,sqrt
import pandas as pd


dataset = load_iris()
dt  = pd.DataFrame(dataset.data,columns=dataset.feature_names)
target =  dataset.target
# cv = LeaveOneOut()
# print(dt.sample(2))
# X_train, X_test, y_train, y_test = train_test_split(dt, target, test_size=0.45, random_state=42)

def KNN(cv,n = 5):
    print(f'KNN: {n}')
    knn = KNeighborsClassifier(n_neighbors=n)
    score = cross_val_score(knn,dt,target,scoring='neg_mean_squared_error',cv=cv,n_jobs=-1)
    return (n, mean(absolute(score)), sqrt(mean(absolute(score))))
def RandomForest(cv,n_trees = 500):
    print("Ramdom Forest")
    rfc =  RandomForestClassifier(n_estimators=n_trees)
    score = cross_val_score(rfc,dt,target,scoring='neg_mean_squared_error',cv=cv,n_jobs=-1)
    return (n_trees, mean(absolute(score)), sqrt(mean(absolute(score))))

if __name__ == '__main__':
    return_metrics = pd.DataFrame(columns=['k','mse','rmse'])
    k =  [1,3,5,7,10]
    cv1 = LeaveOneOut()
    cv2 = 5
    for i in k:
        _k,mse,rmse =  KNN(cv2,i)
        return_metrics.loc[len(return_metrics)] =pd.Series({'k':_k,'mse':mse,'rmse':rmse})
    return_metrics.to_csv('knn_cross_validation_metrics.csv',index=False)
    return_metrics = pd.DataFrame(columns=['k','mse','rmse'])
    for i in k:
        _k,mse,rmse =  KNN(cv1,i)
        return_metrics.loc[len(return_metrics)] =pd.Series({'k':_k,'mse':mse,'rmse':rmse})
    return_metrics.to_csv('knn_leave_one_out_metrics.csv',index=False   )
    return_metrics = pd.DataFrame(columns=['k','mse','rmse'])
    _k,mse,rmse = RandomForest(cv2,500)
    return_metrics.loc[len(return_metrics)] =pd.Series({'n_tree':_k,'mse':mse,'rmse':rmse})
    return_metrics.to_csv('rf_cross_validation_metrics.csv',index=False)
    return_metrics = pd.DataFrame(columns=['k','mse','rmse'])
    _k,mse,rmse = RandomForest(cv2,1000)
    return_metrics.loc[len(return_metrics)] =pd.Series({'n_tree':_k,'mse':mse,'rmse':rmse})
    return_metrics.to_csv('rf_leave_one_out_metrics.csv',index=False   )


# Comparar os resultados em cada tabela
# logo apos comparar com os do artigo da SVM
#acrescentar tecnologias na metodologia
#atualizar a sessão 2.2 -> não citar angiospermas, mas, deixar mais geral
#remove o primeiro paragrafo
#2.3 Estado da Arte -> trabalhos correlatos (similares) "classificação de plantas" 2 a 3 + o atual
#penultimo paragrafo para o atual
# ultimo o nosso trabalho situando o na literatura.