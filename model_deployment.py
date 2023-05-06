import joblib
import zipfile
from flask import Flask, request

app = Flask(__name__)


model = joblib.load("./data_engineer_work_sample_model.joblib")


@app.route('/predict', methods=['GET'])
def predict():
    vol_moving_avg = float(request.args.get('vol_moving_avg'))
    adj_close_rolling_med = float(request.args.get('adj_close_rolling_med'))
    print("Received input parameters: vol_moving_avg={}, adj_close_rolling_med={}".format(vol_moving_avg, adj_close_rolling_med))

    prediction = int(model.predict([[vol_moving_avg, adj_close_rolling_med]]))
    print("Prediction made: {}".format(prediction))

    return str(prediction)


# Start the Flask application
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
