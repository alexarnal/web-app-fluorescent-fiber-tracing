from flask import Flask, render_template, url_for, request, redirect, send_file
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import os 
import shutil
from segmentation.unet_predict import run
import webbrowser
from threading import Timer

def open_browser():
    webbrowser.open_new('http://127.0.0.1:2000/') 

def predict(inFileName, channel, raw_output, thrs_output, contour_output, skeleton_output, unet_agrp, thr):
    print(f"\n\n\n\nRunning Prediction on {inFileName}")
    run(img_path=inFileName, channel=channel, raw_output=raw_output, thrs_output=thrs_output, contour_output=contour_output, skeleton_output=skeleton_output, unet_agrp=unet_agrp, thr=thr)
    os.remove(inFileName)
    print("\n\n\n\nfinished predicting")
    
app= Flask(__name__)
app.config['IMAGE_UPLOADS'] = 'uploads'
app.config['OUTPUT_NAME'] = 'output' # will be overwritten with every file name processed

@app.route('/', methods=['POST','GET']) # GET is default without this method parameter
def index():
    download=False
    multiple = False
    for f in os.listdir(app.config['IMAGE_UPLOADS']):
        os.remove(os.path.join(app.config['IMAGE_UPLOADS'], f))
    if request.method == 'POST':
        if request.files:
            for image in request.files.getlist('image[]'):
                print('getting channel')
                channel = request.form['channel']
                print(channel)
                print('getting output')
                raw_output, thrs_output, contour_output, skeleton_output, unet_agrp = '0', '0', '0', '0', '0'
                try: 
                    unet_agrp = request.form['U-Net_AgRP']
                except: 
                    print('not using unet_agrp')
                try: 
                    raw_output = request.form['raw_output']
                except: 
                    print('not saving raw')
                try: 
                    thrs_output = request.form['thrs_output']
                except: 
                    print('not saving thrs')
                try:
                    contour_output = request.form['contour_output']
                except: 
                    print('not saving contour')
                try:
                    skeleton_output = request.form['skeleton_output']
                except: 
                    print('not saving skeleton')
                thr = float(request.form['threshold'])
                print('threshold is', thr)
                image.save(os.path.join(app.config['IMAGE_UPLOADS'],image.filename))
                app.config['OUTPUT_NAME'] = os.path.splitext(image.filename)[0]
                try:
                    inFileName = os.path.join(app.config['IMAGE_UPLOADS'],image.filename)
                    predict(inFileName, channel, raw_output, thrs_output, contour_output, skeleton_output, unet_agrp, thr)
                    download=True
                except Exception as ex:
                    print('There was an issue at runtime:', ex)
            if len(request.files.getlist('image[]'))>1 or True:
                print('uploaded multiple')
                multiple=True
                shutil.make_archive('download', 'zip', 'uploads')
                return render_template('index.html', download=download, multiple=multiple)
            multiple=False
            return render_template('index.html', download=download, multiple=multiple)
    else:
        return render_template('index.html', download=download, multiple=multiple)
 
@app.route('/download_file')
def download_file():
    p = os.path.join('uploads',app.config['OUTPUT_NAME']+'.svg')
    return send_file(p,as_attachment=True)

@app.route('/download_zip')
def download_zip():
    p = os.path.join('download.zip')
    return send_file(p,as_attachment=True)

@app.route('/')
def again():
    download=False
    multiple=False
    return render_template('index.html', download=download, multiple=multiple)

if __name__ == '__main__':
    Timer(1,open_browser).start(); 
    app.run(port=2000)