import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tkinter as tk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
raw_mail_data = pd.read_csv('https://raw.githubusercontent.com/nimishajames/seminar_MCA/main/mail_data.csv')
root = tk.Tk()

canvas1 = tk.Canvas(root, width=400, height=300, relief='raised')
canvas1.pack()

label1 = tk.Label(root, text='find spam and harm')
label1.config(font=('helvetica', 14))
canvas1.create_window(200, 25, window=label1)

label2 = tk.Label(root, text='Type your mail:')
label2.config(font=('helvetica', 10))
canvas1.create_window(200, 100, window=label2)

entry1 = tk.Entry(root)
canvas1.create_window(200, 140, window=entry1)


def getSquareRoot():
    x1 = entry1.get()


    print(raw_mail_data)
    mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)),'')
    mail_data.head()
    mail_data.shape
    mail_data.loc[mail_data['Category'] == 'spam', 'Category',] = 0
    mail_data.loc[mail_data['Category'] == 'ham', 'Category',] = 1
    X = mail_data['Message']

    Y = mail_data['Category']
    print(X)
    print(Y)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)
    print(X.shape)
    print(X_train.shape)
    print(X_test.shape)

    feature_extraction = TfidfVectorizer(min_df = 1, stop_words='english', lowercase='True')

    X_train_features = feature_extraction.fit_transform(X_train)
    X_test_features = feature_extraction.transform(X_test)

    # convert Y_train and Y_test values as integers

    Y_train = Y_train.astype('int')
    Y_test = Y_test.astype('int')
    print(X_train)
    print(X_train_features)
    model = LogisticRegression()
    model.fit(X_train_features, Y_train)
    prediction_on_training_data = model.predict(X_train_features)
    accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)
    print('Accuracy on training data : ', accuracy_on_training_data)
    input_mail = [x1]

    # convert text to feature vectors
    input_data_features = feature_extraction.transform(input_mail)

    # making prediction

    prediction = model.predict(input_data_features)
    print(prediction)


    if (prediction[0]==1):
        label3 = tk.Label(root, text='Ham', font=('helvetica', 10))
        canvas1.create_window(200, 210, window=label3)

    else:
        label3 = tk.Label(root, text='spam', font=('helvetica', 10))
        canvas1.create_window(200, 210, window=label3)



button1 = tk.Button(text='check mail content', command=getSquareRoot, bg='brown', fg='white',
                    font=('helvetica', 9, 'bold'))
canvas1.create_window(200, 180, window=button1)

root.mainloop()


raw_mail_data.columns = ['Category', 'Message']
raw_mail_data.head()
count_Class = pd.value_counts(raw_mail_data.Category, sort = True)

# Data to Plot
labels = 'Ham', 'Spam'
sizes = [count_Class[0], count_Class[1]]
colors = ['yellow', 'aqua']
explode = (0.1, 0.1)

# Plot
plt.pie(sizes, explode = explode, labels = labels, colors = colors,
        autopct = '%1.1f%%', shadow = True, startangle = 90)
plt.axis('equal')
plt.show()