from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import pandas as pd

app = Flask(__name__, template_folder="Front-end")
CORS(app)

# Load IPL model
try:
    ipl_model = pickle.load(open(r"pipe.pkl", "rb"))
except Exception as e:
    print(f"Error loading IPL model: {e}")
    ipl_model = None

@app.route('/ipl', methods=['POST'])
def predict_ipl():
    if not ipl_model:
        return jsonify({'error': 'IPL Model not loaded'}), 500

    try:
        data = request.get_json(force=True)
        print("✅ Received Data:", data)

        errors = []

        overs = float(data.get('overs', 0))
        wickets = int(data.get('wickets', 0))
        score = int(data.get('score', 0))
        target = int(data.get('target', 0))

        if not (0 < overs <= 20):
            errors.append("Overs must be between 0 and 20")
        if not (0 <= wickets < 10):
            errors.append("Wickets must be between 0 and 9")        
        if errors:
            return jsonify({'errors': errors}), 400
        
        runs_left = target - score
        balls_left = 120 - int(overs * 6)
        wickets_left = 10 - wickets
        crr = score / overs
        rrr = (runs_left * 6 / balls_left) if balls_left > 0 else 0

        input_df = pd.DataFrame({
            'batting_team': [data['batting_team']],
            'bowling_team': [data['bowling_team']],
            'city': [data['city']],
            'runs_left': [runs_left],
            'balls_left': [balls_left],
            'wickets': [wickets],
            'total_runs_x': [target],
            'crr': [crr],
            'rrr': [rrr]
        })

        print("📊 Input DF:", input_df)

        result = ipl_model.predict_proba(input_df)
        print("✅ Prediction result:", result)

        return jsonify({
            'win_prob': round(result[0][1] * 100, 2),
            'loss_prob': round(result[0][0] * 100, 2)
        })

    except Exception as e:
        print("🔥 Exception:", e)
        return jsonify({'error': str(e)}), 500

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
