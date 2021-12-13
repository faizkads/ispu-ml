#import Flask
from flask import Flask, render_template, request
import joblib
import numpy as np
from keras.models import load_model

#create an instance of Flask
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict/', methods=['GET','POST'])
def predict():
    
    if request.method == "POST":
        
        #get form data
        day_1 = request.form.get('day_1')
        day_2 = request.form.get('day_2')
        day_3 = request.form.get('day_3')
        day_4 = request.form.get('day_4')
        day_5 = request.form.get('day_5')
        
        #call preprocessDataAndPredict and pass inputs
    try:
        prediction = preprocessDataAndPredict(day_1,day_2, day_3, day_4, day_5)
        ispu_info = classify(prediction)

        #pass prediction to template
        return render_template('predict.html', prediction = prediction, ispu_info = ispu_info)
   
    except ValueError:
        return "Please Enter valid values"
  
        pass
    pass

def preprocessDataAndPredict(day_1,day_2, day_3, day_4, day_5):

    #load model and scaler
    model = load_model("model.h5")
    scaler = joblib.load(open('scaler.pkl', 'rb'))
    
    print(day_1,day_2, day_3, day_4, day_5)

    #keep all inputs in array
    test_data = np.array([[day_1],[day_2],[day_3],[day_4],[day_5]])
    print(test_data)

    test_data = scaler.transform(test_data)
    print(test_data)

    #reshape array
    test_data = test_data.reshape(1,-1, 1)
    print(test_data)
    
    #predict
    prediction = model.predict(test_data)
    prediction = scaler.inverse_transform(prediction)
    prediction = np.asscalar(prediction)
    
    return prediction

def classify(prediction):
    if prediction <= 50:
        ispu_info = 'Baik'
    elif prediction >= 51 and prediction <= 100:
        ispu_info = 'Sedang'
    elif prediction >= 101 and prediction <= 199:
        ispu_info = 'Tidak Sehat'
    elif prediction >= 201 and prediction <= 299:
        ispu_info = 'Sangat Tidak Sehat'
    elif prediction >= 300:
        ispu_info = 'Berbahaya'
    print(ispu_info)
    return ispu_info
    
if __name__ == '__main__':
    app.run(debug=True)