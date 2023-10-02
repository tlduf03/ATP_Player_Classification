from flask import Flask, request, render_template, jsonify
import util

app = Flask(__name__)

@app.route('/classify_image', methods=['GET','POST'])
def classify_image():
    return 'Hello'

if __name__ == "__main__":
    # util.load_saved_artifacts()
    app.run(port=5000, debug=True)