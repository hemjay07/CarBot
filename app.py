from flask import Flask, render_template, jsonify, request, send_from_directory
import processor
#from  skimage import io
import os
from werkzeug.utils import secure_filename



app = Flask(__name__)

app.config['SECRET_KEY'] = 'enter-a-very-secretive-key-3479373'


@app.route('/', methods=["GET", "POST"])
def index():
    return render_template('index.html', **locals())


@app.route('/chat', methods=["GET", "POST"])
def chat():
	return render_template('chat.html')


@app.route('/chatbot', methods=["GET", "POST"])
def chatbotResponse():

    if request.method == 'POST':
        the_question = request.form['question']

        response = processor.chatbot_response(the_question)

    return jsonify({"response": response })

@app.route('/image', methods=["GET", "POST"])
def upload():
    if request.method == 'POST':
        f = request.files['image_question']
        basepath = os.path.abspath(os.path.dirname(__file__))
        img_path =os.path.join(basepath,'uploads', secure_filename(f.filename))
        f.save(img_path)
        
       # question= io.imread(the_question)
    
        img_response = processor.image_response(img_path)

    return jsonify({"img_response": img_response})
	
@app.route('/image/<filename>')
def send_file(filename):
    return send_from_directory('uploads', filename)



if __name__ == '__main__':
    app.run()
