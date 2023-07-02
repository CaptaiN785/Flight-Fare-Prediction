from flask import (
    Flask,
    render_template,
    jsonify,
    request,
    redirect,
    url_for
)
import requests
import os
import zipfile
import pandas as pd
import pickle
import numpy as np
import sklearn

DEBUG = True

if DEBUG:
    os.environ.setdefault("ACCESS_TOKEN", open('var/access_token.txt').read())

#################### Processing ########################

MODEL_LINK = "https://firebasestorage.googleapis.com/v0/b/model-up.appspot.com/o/models.zip?alt=media&token={}".format(os.environ.get("ACCESS_TOKEN"))
model_zip_path = "model/models.zip"
ecnomic_model_path = "model/economy_model.pkl"
business_model_path = "model/business_model.pkl"

if not os.path.exists("model"):
    os.mkdir("model")

if not os.path.exists(model_zip_path):
    print("Downloading model zip file...")
    data = requests.get(MODEL_LINK)
    with open(model_zip_path, "wb") as zip:
        zip.write(data.content)

if not os.path.exists(ecnomic_model_path) or not os.path.exists(business_model_path):
    print("Extracting the models...")
    with zipfile.ZipFile(model_zip_path) as zip:
        zip.extractall("model")


categorical = ['airline', 'flight', 'source_city', 'departure_time', 'stops','arrival_time','destination_city']
numerical = ['duration', 'days_left']

ec = pd.read_csv('economy.csv')
bs = pd.read_csv('business.csv')

city_list = ec['source_city'].unique()
airlines  = ec['airline'].unique()
economy_flights = sorted(ec['flight'].unique())
business_flights = sorted(bs['flight'].unique())
departure_times = ec['departure_time'].unique()
stops = ec['stops'].unique()

columns = ec.columns[:-1]

def convert_to_minutes(x):
    arr = x.split(":")
    return float(arr[0])*60 + float(arr[1])

def transform_data(col, data):
    encoder = pickle.load(open("encoder.pkl", "rb"))
    encoded_data = encoder[col].transform(data)
    return encoded_data

def predict_price(df, class_):
    if class_ == 'economy':
        model = pickle.load(open(ecnomic_model_path, "rb"))
    else:
        model = pickle.load(open(business_model_path, "rb"))
    
    pred = model.predict(df)
    return round(pred[0])


###########################################################


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', \
        cities=city_list,\
        airlines = airlines,\
        departure_times = departure_times,\
        stops = stops,\
        )

## List of flights which is in bussiness class.
@app.route('/getflights/<classtype>', methods= ['GET'])
def get_flights(classtype):
    if classtype == 'economy':
        return jsonify({"flights": economy_flights})
    else:
        return jsonify({"flights": business_flights})


@app.route("/predict", methods=["POST", "GET"])
def predict():
    if request.method == "GET":
        return redirect('/')

    ## Reading the data from form
    try:
        if request.method == "POST":
            input_data = {}
            for col in categorical:
                input_data[col] = transform_data(col, [request.form[col]])
            for col in numerical:
                input_data[col] = [request.form[col]]
            
            ## Converting form data to dataframe
            input_df = pd.DataFrame(input_data)

            ## Converting the duration to minute from HH:MM to minute
            input_df['duration'] = input_df['duration'].apply(lambda x: convert_to_minutes(x))

            # print(input_df)
            class_ = request.form["class"]
            ## Check for class
            fare = predict_price(input_df, class_)
        
        return render_template('result.html', fare = fare)
    except:
        return redirect('/')


if __name__ == '__main__':
    app.run(debug=DEBUG)