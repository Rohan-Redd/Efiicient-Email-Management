import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# load the data
df = pd.read_excel("C:/Users/Adavelli Rohan Reddy/Desktop/VIT/6th sem/RPA/Project/Email_Auto_modern/Email/Departments/Depts.xlsx", engine="openpyxl")

# create a TF-IDF vectorizer object
tfidf = TfidfVectorizer(max_features=10000)

# transform the text data into numerical feature vectors
X = tfidf.fit_transform(df['Query'])

# create the target variable
y = df['Department']

# create a Naive Bayes classifier object
clf = MultinomialNB()

# fit the classifier to the data
clf.fit(X, y)

# save the model to a .sav file using pickle
with open("model.sav", "wb") as f:
    pickle.dump(clf, f)

# create a function to predict the department
"""def predict_department(text):
    with open("model.sav", "rb") as f:
        clf = pickle.load(f)
    X_test = tfidf.transform([text])
    return clf.predict(X_test)[0]"""

input = "C:/Users/Adavelli Rohan Reddy/Desktop/VIT/6th sem/RPA/Project/Email_Auto_modern/Email/Departments/inp.txt"
output = "C:/Users/Adavelli Rohan Reddy/Desktop/VIT/6th sem/RPA/Project/Email_Auto_modern/Email/Departments/out.txt"

def predict_department(input_file, output_file):
    with open("model.sav", "rb") as f:
        clf = pickle.load(f)
    # read the input text file
    with open(input_file, 'r') as f:
        input_text = f.read()
    X_test = tfidf.transform([input_text])
    # predict the department
    predicted_department = clf.predict(X_test)[0]
    # write the predicted department to the output file
    with open(output_file, 'w') as f:
        f.write(predicted_department)
predict_department(input,output)

#a = predict_department("Dear Admissions Office,I am writing to express my interest in applying to [College Name] for the upcoming academic year. I am excited about the possibility of studying at your institution and would like to request more information about the application process.I am particularly interested in the [major/program] offered by [College Name] and believe that the curriculum and resources provided by your institution would allow me to achieve my academic and professional goals. I have researched [College Name] extensively and am impressed by the achievements and reputation of its faculty and students.Could you please provide me with more information on the following:Application requirements and deadlinesRequired documents and transcriptsFinancial aid and scholarship opportunitiesCampus facilities and student resourcesAny other relevant information about [College Name]Thank you for your time and consideration. I look forward to hearing from you soon.")
#print(a)

#b = predict_department("I am writing to inquire about housing options available for students at [College Name] for the upcoming academic year. As I am planning to attend [College Name], I am interested in exploring the various housing options that the institution has to offer.Could you please provide me with more information on the following:Types of housing available (e.g. dormitories, apartments, off-campus housing)Availability of on-campus housing for freshmen or transfer studentsRoommate matching processHousing costs and payment optionsMeal plan options and costsAny other relevant information about housing at [College Name]I would greatly appreciate it if you could provide me with any additional information that would help me in making an informed decision about my housing options at [College Name]. Thank you for your time and consideration.")
#print(b)