from flask import Flask, render_template, jsonify, request
import requests
import os
import zipfile
import pandas as pd
import pickle

#################### Processing.....

MODEL_LINK = "https://storage.googleapis.com/cushare-785.appspot.com/captain/2023-06-30%2009%3A49%3A19.522027%20Fligh%20Fare%20prediction%20model/models.zip"
model_zip_path = "model/models.zip"
ecnomic_model_path = "model/economy_model.pkl"
business_model_path = "model/business_model.pkl"


if not os.path.exists(model_zip_path):
    data = requests.get(MODEL_LINK)
    with open(model_zip_path, "wb") as zip:
        zip.write(data.content)

if not os.path.exists(ecnomic_model_path) or not os.path.exists(business_model_path):
    with zipfile.ZipFile(model_zip_path) as zip:
        zip.extractall("model")


## Loading the encoder
encoder = pickle.load(open("encoder.pkl", "rb"))
categorical = ['airline', 'flight', 'source_city', 'departure_time', 'stops','arrival_time','destination_city']
numerical = ['duration', 'days_left', 'price']

ec = pd.read_csv('economy.csv')
bs = pd.read_csv('business.csv')

print(ec.columns)
print(ec['duration'].max())

city_list = ec['source_city'].unique()
airlines  = ec['airline'].unique()
economy_flights = sorted(ec['flight'].unique())
business_flights = sorted(bs['flight'].unique())
departure_times = ec['departure_time'].unique()
stops = ec['stops'].unique()


###########################################################

## Format
# Src city - Dest city
# airline  - flight
# departure time - arrival time
# stops - duration
# day left - button



app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html', \
        cities=city_list,\
        airlines = airlines,\
        departure_times = departure_times,\
        stops = stops,\
        )

@app.route('/getflights/<classtype>', methods= ['GET'])
def get_flights(classtype):
    if classtype == 'economy':
        return jsonify({"flights": economy_flights})
    else:
        return jsonify({"flights": business_flights})

if __name__ == '__main__':
    app.run(debug=True)