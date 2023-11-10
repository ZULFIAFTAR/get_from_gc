from flask import Flask, render_template, request
import numpy as np
import pickle
import pandas as pd

app = Flask(__name__)

# Load the model
loaded_model = pickle.load(open('clf_model.pkl','rb'))
loaded_vectorizer = pickle.load(open('vectorizer.pickle','rb'))


@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict2', methods=['POST'])
def predict2():
    data_sql_statement = request.form['sql_statement']
    
    arr = np.array([data_sql_statement])
    pred = loaded_model.predict(loaded_vectorizer.transform(arr.ravel()))

    predicted_output = pd.DataFrame(pred, columns=['Result'])
    return render_template("result.html",
                           tables=[predicted_output.to_html(classes='data', index=False)],
                           titles=predicted_output.columns.values)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80, debug=True)
