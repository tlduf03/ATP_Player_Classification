from flask import Flask, request, jsonify
import util


app = Flask(__name__)

@app.route('/classify_image', methods=['GET', 'POST'])
def classify_image():
    #this function will be used to classify images using the trained model
    image_data = request.form['image_data']
    response = jsonify(util.classify_images(image_data))
    
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

if __name__ == "__main__":
    util.load_saved_artifacts()
    app.run(port=5000, debug=True)