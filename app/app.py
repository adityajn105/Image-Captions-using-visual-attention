"""
By - Aditya Jain
Contact - https://adityajain.me
Last Update - 9 April 2020
"""

from flask import Flask, render_template, request, jsonify, redirect
import base64
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
	if request.method == 'POST':
		image=request.files['image']
		image_string = base64.b64encode(image.read())
		
		binary = base64.b64decode(image_string)
		img = np.asarray(bytearray(binary), dtype="uint8")
		img = cv2.imdecode(img, 1)
		img = cv2.resize( img, (224,224), interpolation=cv2.INTER_AREA )
		
		return render_template('index.html', data={ 'status':True, 'img': str(image_string)[2:-1], 'caption': "This is app is under training progress."})
	else:
		return render_template('index.html', data={ 'status':False })

if __name__== '__main__':
    app.run(host = '0.0.0.0', debug=True, port = int(5000))