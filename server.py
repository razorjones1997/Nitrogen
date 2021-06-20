import pickle
import numpy as np
from flask import Flask, render_template, request
app = Flask(__name__)

@app.route('/', methods=['POST', 'GET']) # This is the root route
def root():
    if request.method == "POST":
        ph = float(request.form.get("pH"))
        OC = float(request.form.get("Organic Carbon"))
        EC = float(request.form.get("Electric Conductivity"))
        P = float(request.form.get("Phosphorus"))
        K = float(request.form.get("Potassium"))
        S = float(request.form.get("Sulphur"))
        Zn = float(request.form.get("Zinc"))
        Fe = float(request.form.get("Iron"))
        Cu = float(request.form.get("Copper"))
        Mn = float(request.form.get("Manganese"))
        B = float(request.form.get("Boron"))
        
        loaded_model = pickle.load(open('models/HardVotingClassifierModel_Nitrogen.pkl', 'rb'))
        test_li = np.array([ph, OC, EC, P, K, S, Zn, Fe, Cu, Mn, B]).reshape(1, -1)
        lp = loaded_model.predict(test_li)

        return render_template("result.html", val=lp)
    return render_template('index.html') # render_template sends the HTML file to the browser
