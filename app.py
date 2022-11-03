import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)

#Loaded the model and scalar
model=pickle.load(open('reg.pkl','rb'))
scalar=pickle.load(open('scalar.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    scaled_data=scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output=model.predict(scaled_data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=scalar.transform([data]).reshape(1,-1)
    output=model.predict(final_input)[0]
    return render_template('index.html',prediction_text=f'The predicted House Price is {output:.2f}.')

if __name__ == "__main__":
    app.run(debug=True)