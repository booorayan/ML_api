# ML Model as a web service
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from joblib import dump


df = pd.read_csv('/home/bo/Downloads/penguins_size.csv')

# Drop null values
df.dropna(inplace=True)

# Drop island column
df.drop('island', axis=1, inplace=True)


# Label encoding
enc = LabelEncoder()

for col in df.select_dtypes(include='object'):
    df[col] = enc.fit_transform(df[col])


# Define train and test sets
y = df.species

df.drop('species', axis=1, inplace=True)
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.15)


# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


# Validate results
acc = accuracy_score(y_pred, y_test)
print(f'The accuracy of the model is {acc}')


# Save model
dump(model, 'penguin_model')
