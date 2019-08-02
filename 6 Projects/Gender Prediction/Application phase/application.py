from flask import Flask,render_template,request
import pandas as pd
import numpy
import pickle
from sklearn.preprocessing import LabelEncoder
app = Flask(__name__)

@app.route("/",methods=["POST","GET"])
def index():
	if request.method == "POST":
		f = open('trainedModelfile','rb')
		trained_model = pickle.load(f)
		f.close()
		encoder_class =numpy.load('Encoder_class.npy').all()
		height =request.form.get('Height')
		weight = request.form.get('Weight')
		hair = request.form.get('Hair')
		beard = request.form.get('Beard')
		scarf =request.form.get('Scarf')

		input_feature_vector = pd.DataFrame({'height':[round(float(height),1)],
                                     'weight':[int(weight)],
                                    'hair':[hair],
                                     'beard':[beard],
                                     'scarf':[scarf]
                                    })

		Encoded_input_feature_vector = input_feature_vector.copy()
		
		for c in Encoded_input_feature_vector.iloc[:,2:].columns:
    			Encoded_input_feature_vector[c] = encoder_class[c].transform(Encoded_input_feature_vector[c])
		

		prediction_of_user_input = trained_model.predict(Encoded_input_feature_vector)
		decoded_prediction_of_user_input = encoder_class['gender'].inverse_transform(prediction_of_user_input)[0]
		return render_template("index.html",prediction = decoded_prediction_of_user_input)

	return render_template("index.html")	