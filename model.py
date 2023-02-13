# ML Model as a web service
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from joblib import dump


# Load dataset
df = pd.read_csv('/home/bo/Downloads/penguins_size.csv')

# Drop null values permanently
df.dropna(inplace=True)

# Drop island column
df.drop('island', axis=1, inplace=True)


# Label encoding to convert categorical columns to numerical values
enc = LabelEncoder()

for col in df.select_dtypes(include='object'):
    df[col] = enc.fit_transform(df[col])


# Define dependent variable
y = df.species

# Define train and test sets
df.drop('species', axis=1, inplace=True)
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.15)


# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)


# Validate results
acc = accuracy_score(y_pred, y_test)
print(f'The accuracy of the model is {acc}')

# Performance evaluation
print(f'Model score with training data is: \n{model.score(X_train,y_train)}')
print(f'Model score with test data is: \n{model.score(X_test, y_test)}')
print(f'\nGiven the model score with test data is {model.score(X_test, y_test)} \
which is not a huge deviation from the score with training data, \
i.e, {model.score(X_train,y_train)}, we can conclude the model is not overfitting!!')

# Using confusion_matrix()
print('\nFurther performance evaluation...')
print(f'The confusion matrix is: \n{confusion_matrix(y_test, y_pred, labels=[1,2,3])}')

# Check for important features
print(f'Important features: \n{pd.DataFrame(model.feature_importances_, index=df.columns).sort_values(by=0, ascending=False)}')

# Save model
dump(model, 'penguin_model')
