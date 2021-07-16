# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 17:01:05 2021

@author: aishw
"""

from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)

model=pickle.load(open('model.pkl','rb'))


@app.route('/')
def hello_world():
    return render_template("app.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features=[int(x) for x in request.form.values()]
    final=[np.array(int_features)]
    print(int_features)
    print(final)
    prediction=model.predict_proba(final)
    output='{0:.{1}f}'.format(prediction[0][1], 2)

    if output>str(0.5):
        return render_template('app.html',pred='Your customer is MORE LIKELY to churn.\nProbability of churning is {}'.format(output),bhai="Take preventive measures")
    else:
        return render_template('app.html',pred='Your customer is NOT LIKELY to churn.\n Probability of churning is {}'.format(output),bhai="Customer is happy with the service")


if __name__ == '__main__':
    app.run(debug=True)