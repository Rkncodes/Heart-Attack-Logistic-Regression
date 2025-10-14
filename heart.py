import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
#dataset
df = pd.read_csv("Medicaldataset.csv")
#Check and print null values
print("Null values before cleaning:\n", df.isnull().sum())
#there is no null values in the data set
#Result as bool
le = LabelEncoder()
df['Result'] = le.fit_transform(df['Result'])
x = df.drop("Result", axis=1)
y = df["Result"]
#Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
#Train logistic regression model
log_model = LogisticRegression(max_iter=1000)
log_model.fit(x_train, y_train)
#Predictions and metrics
y_pred = log_model.predict(x_test)
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy Score:", round(accuracy * 100, 2), "%")
#Heatmap of correlation
sns.heatmap(conf_matrix, annot=True,)
plt.title("Feature Correlation Heatmap")
plt.show()
