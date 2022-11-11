from multiprocessing.spawn import import_main_path
import pandas as pd
import matplotlib.pyplot as plt
from django.conf import settings
from sklearn.tree import DecisionTreeClassifier
 

class Alprocess:
    path = settings.MEDIA_ROOT + "\\" + "scholar.csv"
    data=pd.read_csv(path,delimiter=',')

    x = data.iloc[:,0:5]
    y = data.iloc[:,-1]

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=1)
    
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    # x_train = sc.fit_transform(x_train)
    # x_test = sc.fit_transform(x_test)

    def Proces_Decision_tree(self):
        #Fitting Decision Tree classifier to the training set  
        from sklearn.tree import DecisionTreeClassifier
        decisiontree = DecisionTreeClassifier()

        decisiontree.fit(self.x_train, self.y_train)
        ypred = decisiontree.predict(self.x_test)
        
        import numpy as np
        import seaborn as sns
        sns.distplot(self.data)
        plt.show()
        sns.countplot(self.data["percentage"])
        plt.show()
        sns.heatmap(self.data)
        plt.show()
        sns.pairplot(self.data)
        plt.show()
        # performance metrics
        from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,f1_score,recall_score
        
        y_test = self.y_test.astype(int)
        accuracy =accuracy_score(y_test,ypred)
        precission = precision_score(y_test,ypred,average='weighted')
        f1= f1_score(y_test, ypred,average='weighted')
        recall =recall_score(y_test, ypred,average='weighted')
        return accuracy, precission, f1, recall

    def Knn(self):
        from sklearn.neighbors import KNeighborsClassifier
        neigh = KNeighborsClassifier(n_neighbors=3)

        neigh.fit(self.x_train, self.y_train)
        ypred = neigh.predict(self.x_test)
        
        import numpy as np
        import plotly.graph_objects as go
        import numpy as np
        from sklearn.datasets import make_moons
        from sklearn.model_selection import train_test_split
        from sklearn.neighbors import KNeighborsClassifier

        # performance metrics
        from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,f1_score,recall_score
        
        y_test = self.y_test.astype(int)
        accuracy =accuracy_score(y_test,ypred)
        precission = precision_score(y_test,ypred,average='weighted')
        f1= f1_score(y_test, ypred,average='weighted')
        recall =recall_score(y_test, ypred,average='weighted')
        print('acc:',accuracy)
        return accuracy, precission, f1, recall

    def NaiveBayes(self):
        from sklearn.naive_bayes import GaussianNB
        clf = GaussianNB()


        clf.fit(self.x_train, self.y_train)
        ypred = clf.predict(self.x_test)
        
        import numpy as np
        
    
        # performance metrics
        from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,f1_score,recall_score
        
        y_test = self.y_test.astype(int)
        accuracy =accuracy_score(y_test,ypred)
        precission = precision_score(y_test,ypred,average='weighted')
        f1= f1_score(y_test, ypred,average='weighted')
        recall =recall_score(y_test, ypred,average='weighted')
        print('acc:',accuracy)
        return accuracy, precission, f1, recall



         