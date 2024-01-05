# Importing libraries
import pandas as pd
import pickle
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
port_stem=PorterStemmer()

# Load dataset
df = pd.read_csv("dataset.csv")
df = df.fillna('')
df = df.drop(['id', 'title', 'author'], axis=1)

# Preprocessing functions
def stemming(content):
    con=re.sub('[^a-zA-Z]', ' ', content)
    con=con.lower()
    con=con.split()
    con=[port_stem.stem(word) for word in con if not word in stopwords.words('english')]
    con=' '.join(con)
    return con
nltk.download('stopwords')
df['text']=df['text'].apply(stemming)

# Train-test split
x = df['text']
y = df['label']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

# Vectorization
vect = TfidfVectorizer()
x_train_vect = vect.fit_transform(x_train)
x_test_vect = vect.transform(x_test)

# Train models
models = {
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(),
    'Multinomial Naive Bayes': MultinomialNB()
}

for name, model in models.items():
    model.fit(x_train_vect, y_train)
    y_pred = model.predict(x_test_vect)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy}")

# Pickle models
for name, model in models.items():
    pickle.dump(model, open(f'{name.lower().replace(" ", "_")}_model.pkl', 'wb'))
