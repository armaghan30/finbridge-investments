from flask import Flask
import os

app = Flask(__name__)

@app.route('/')
def home():
    return '<h1>FinBridge is Working!</h1><p>Railway deployment successful!</p>'

@app.route('/health')
def health():
    return {'status': 'ok', 'message': 'App is running'}

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)