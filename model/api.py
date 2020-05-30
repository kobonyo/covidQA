from flask import Flask
from flask import request
from models import Models
model = Models()
app = Flask(__name__)
@app.route('/api')
def index():
    q = request.args.get('q')
    return model.predict(q)

if __name__ == '__main__':
    app.run(debug = True )