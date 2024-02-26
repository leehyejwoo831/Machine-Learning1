from flask import Flask, render_template, request
import pickle
import numpy as np

# 모델 로드
model = pickle.load(open('car.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def man():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def home():
    data1 = int (request.form['a'])
    data2 = int (request.form['b'])
    data3 = int (request.form['c'])
    data4 = float (request.form['d'])
    data5 = int (request.form['e'])
    arr = np.array([[data1, data2, data3, data4, data5]])
    pred = model.predict(arr)
    return render_template('home.html', data=pred)

if __name__ == "__main__":
    app.run(debug=True)