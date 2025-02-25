import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import pickle
experience = np.random.randint(0, 20,20)
test_score = np.random.randint(6,11,20)
interview_score = np.random.randint(6,11,20)
salary = np.random.randint(50000, 80001,20)
df = pd.DataFrame({'experience': experience, 'test_score': test_score, 'interview_score': interview_score, 'salary': salary})
print(df)
X = df.iloc[:,:3]
Y= df.iloc[:,-1]
print(X)
print(Y)
regressor = LinearRegression()
regressor.fit(X,Y)
print(regressor.coef_)

pickle.dump(regressor, open('model.pkl','wb'))
model =pickle.load(open('model.pkl','rb'))
print(model.predict([[4,9,9]]))

print(np.__version__)
print(pd.__version__)
print(sklearn.__version__)