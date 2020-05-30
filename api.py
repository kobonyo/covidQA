from flask import Flask, jsonify
from flask import request
from models import Models
model = Models()
app = Flask(__name__)

@app.route('/api')
def index(methods=['GET']):
    q = request.args.get('q')
    res = {       
        'answer': "model.predict(q)"
    }
    return jsonify(res)

if __name__ == '__main__':
    app.run(debug = True )