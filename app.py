from flask import Flask, request, jsonify
from flask_cors import CORS
from resolve import resolve_conflict

app = Flask(__name__)
CORS(app)

@app.route('/resolve', methods=['POST'])
def resolve():
    data = request.get_json()
    base = data.get('base')
    local = data.get('local')
    remote = data.get('remote')

    if not base or not local or not remote:
        return jsonify({'error': 'Missing input'}), 400

    try:
        result = resolve_conflict(base, local, remote)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
