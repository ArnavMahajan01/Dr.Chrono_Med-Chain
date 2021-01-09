
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np

import sys

from keras.models import load_model
model = load_model('chatbot_model.h5')
import json
import random
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))


import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier,_tree
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.tree import export_graphviz
import warnings
import sys
warnings.filterwarnings("ignore", category=DeprecationWarning)

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

# def symptom_disease(theSym):
#     training = pd.read_csv('Training.csv')
#     testing  = pd.read_csv('Testing.csv')
#     cols     = training.columns
#     cols     = cols[:-1]
#     x        = training[cols]
#     y        = training['prognosis']
#     y1       = y

#     reduced_data = training.groupby(training['prognosis']).max()

#     #mapping strings to numbers
#     le1 = preprocessing.LabelEncoder()
#     le = le1.fit(y)
#     y = le.transform(y)




#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
#     testx    = testing[cols]
#     testy    = testing['prognosis']  
#     testy    = le.transform(testy)


#     clf1  = DecisionTreeClassifier()
#     clf = clf1.fit(x_train,y_train)

#     importances = clf.feature_importances_
#     indices = np.argsort(importances)[::-1]
#     features = cols

#     q = ""

#     def print_disease(node):
#         #print(node)
#         node = node[0]
#         #print(len(node))
#         val  = node.nonzero() 
#         #print(val)
#         disease = le.inverse_transform(val[0])
#         return disease
#     def tree_to_code(tree, feature_names):
#         tree_ = tree.tree_
#         feature_name = [
#             feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
#             for i in tree_.feature
#         ]
#         #print("def tree({}):".format(", ".join(feature_names)))
#         symptoms_present = []
#         def recurse(node, depth):
#             indent = "  " * depth
#             if tree_.feature[node] != _tree.TREE_UNDEFINED:
#                 name = feature_name[node]
#                 threshold = tree_.threshold[node]

#                 val = 0
#                 for si in theSym:
#                     if si == name:
#                         val = 1
                
#                 if  val <= threshold:
#                     return recurse(tree_.children_left[node], depth + 1)
#                 else:
#                     symptoms_present.append(name)
#                     return recurse(tree_.children_right[node], depth + 1)
#             else:
#                 present_disease = print_disease(tree_.value[node])
#                 q = "You may have " +  present_disease[0] 
#                 red_cols = reduced_data.columns 
#                 symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]
#                 # print("symptoms present  " + str(list(symptoms_present)))
#                 # print("symptoms given "  +  str(list(symptoms_given)) )  
#                 # confidence_level = (1.0*len(symptoms_present))/len(symptoms_given)
#                 # print("confidence level is " + str(confidence_level))
#             return q
            

#         return recurse(0, 1)

#     return tree_to_code(clf,cols)

def symptom_disease(theSym):
    training = pd.read_csv('Training.csv')
    testing  = pd.read_csv('Testing.csv')
    cols     = training.columns
    cols     = cols[:-1]
    x        = training[cols]
    y        = training['prognosis']
    y1       = y

    reduced_data = training.groupby(training['prognosis']).max()

    #mapping strings to numbers
    le1 = preprocessing.LabelEncoder()
    le = le1.fit(y)
    y = le.transform(y)




    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
    testx    = testing[cols]
    testy    = testing['prognosis']  
    testy    = le.transform(testy)


    clf1  = DecisionTreeClassifier()
    clf = clf1.fit(x_train,y_train)

    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    features = cols

    def print_disease(node):
        #print(node)
        node = node[0]
        #print(len(node))
        val  = node.nonzero() 
        #print(val)
        disease = le.inverse_transform(val[0])
        return disease
    def tree_to_code(tree, feature_names):
        tree_ = tree.tree_
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]
        #print("def tree({}):".format(", ".join(feature_names)))
        symptoms_present = []
        def recurse(node, depth):
            indent = "  " * depth
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                threshold = tree_.threshold[node]

                val = 0
                for si in theSym:
                    if si == name:
                        val = 1
                
                if  val <= threshold:
                    recurse(tree_.children_left[node], depth + 1)
                else:
                    symptoms_present.append(name)
                    recurse(tree_.children_right[node], depth + 1)
            else:
                present_disease = print_disease(tree_.value[node])
                dater = ""
                for z in present_disease:
                    dater += z + ", "
                print("You may have " +  dater) 
            

        recurse(0, 1)

    tree_to_code(clf,cols)
    

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res


mg = sys.argv[1]
i = predict_class(mg, model)
if i[0]["intent"] != "disease_symptoms":
    print(chatbot_response(mg))
else:
    symps = mg.split(',')
    sym = list()
    for s in symps:
        if s.startswith(' ') :
            s = s[1:]
        if s.endswith(' '):
            s = s[:-2]
        s = s.replace(' ', '_')
        sym.append(s)
    symptom_disease(sym)

# def response(mg):
#     res = ""
#     i = predict_class(mg, model)
#     if i[0]["intent"] != "disease_symptoms":
#         return chatbot_response(mg)
#     else:
#         symps = mg.split(',')
#         sym = list()
#         for s in symps:
#             if s.startswith(' ') :
#                 s = s[1:]
#             if s.endswith(' '):
#                 s = s[:-2]
#             s = s.replace(' ', '_')
#             sym.append(s)
#         return symptom_disease(sym)

       

        



#Creating GUI with tkinter
# import tkinter
# from tkinter import *


# def send():
#     msg = EntryBox.get("1.0",'end-1c').strip()
#     EntryBox.delete("0.0",END)

#     if msg != '':
#         ChatLog.config(state=NORMAL)
#         ChatLog.insert(END, "You: " + msg + '\n\n')
#         ChatLog.config(foreground="#442265", font=("Verdana", 12 ))

#         res = response(msg)
#         ChatLog.insert(END, "Bot: " + res + '\n\n')

#         ChatLog.config(state=DISABLED)
#         ChatLog.yview(END)


# base = Tk()
# base.title("Hello")
# base.geometry("400x500")
# base.resizable(width=FALSE, height=FALSE)

# #Create Chat window
# ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Arial",)

# ChatLog.config(state=DISABLED)

# #Bind scrollbar to Chat window
# scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
# ChatLog['yscrollcommand'] = scrollbar.set

# #Create Button to send message
# SendButton = Button(base, font=("Verdana",12,'bold'), text="Send", width="12", height=5,
#                     bd=0, bg="#32de97", activebackground="#3c9d9b",fg='#ffffff',
#                     command= send )

# #Create the box to enter message
# EntryBox = Text(base, bd=0, bg="white",width="29", height="5", font="Arial")
# #EntryBox.bind("<Return>", send)


# #Place all components on the screen
# scrollbar.place(x=376,y=6, height=386)
# ChatLog.place(x=6,y=6, height=386, width=370)
# EntryBox.place(x=128, y=401, height=90, width=265)
# SendButton.place(x=6, y=401, height=90)

# base.mainloop()