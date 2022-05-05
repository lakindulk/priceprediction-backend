import json
from this import d
from unicodedata import name
import pandas as pd
from flask import Flask,render_template,request
import pickle
import numpy as np
from flask_cors import CORS 

app=Flask(__name__)
CORS(app)
data=pd.read_csv('Cleaned_data.csv')
pipe=pickle.load(open("RidgeModel1.pkl" ,'rb'))

@app.route('/city',methods=['GET'])
def index():
    locations=sorted(data['Location'].unique())
    country_list=[]
    for row in locations:
            country_list.append(row)
    return json.dumps(country_list)


@app.route('/predict',methods=['POST'])
def predict():
    #In get data use name in bootstrap form
    location=request.json.get('location')
    bhk=request.json.get('bed')
    bath=request.json.get('bath')
    sqft=request.json.get('total_sqft')
  
  
    
    beds=float(bhk)
    baths=float(bath)
    total_sqft=float(sqft)
    print(beds)
    print(baths)
    print(total_sqft)

    input=pd.DataFrame([[baths,beds,total_sqft,location]],columns=["Baths","Beds","House size","Location"])

    prediction=pipe.predict(input)[0]
    print(prediction)
    return json.dumps({ "result": np.round(prediction,2) })

if __name__ == '__main__':
   app.run(debug=True,port=5001)