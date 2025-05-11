from flask import Flask, request, jsonify
from flask import Flask, render_template
from flask_cors import CORS
import pickle
import pandas as pd

app = Flask(__name__, template_folder="Front-end") 
CORS(app, resources={r"/*": {"origins": "*"}})

# Load T20 model
try:
    t20_model = pickle.load(open(r"T20/t20model.pkl", "rb"))
except Exception as e:
    print(f"Error loading T20 model: {e}")
    t20_model = None

@app.route('/predict/t20', methods=['POST'])
def predict_t20():
    if not t20_model:
        return jsonify({'error': 'T20 Model not loaded'}), 500

    try:
        data = request.get_json(force=True)
        print("✅ T20 Data Received:", data)

        errors = []

        overs = float(data.get('overs', 0))
        wickets = int(data.get('wickets', 0))
        score = int(data.get('score', 0))
        target = int(data.get('target', 0))

        # Validation for T20 (20 overs max)
        if not (0 < overs <= 19):
            errors.append("Overs must be between 0 and 20")
        if not (0 <= wickets < 9):
            errors.append("Wickets must be between 0 and 9")

        if errors:
            return jsonify({'errors': errors}), 400

        runs_left = target - score
        balls_left = 120 - int(overs * 6)
        crr = score / overs
        rrr = (runs_left * 6 / balls_left) if balls_left > 0 else 0

        input_df = pd.DataFrame({
            'batting_team': [data['batting_team']],
            'bowling_team': [data['bowling_team']],
            'city': [data['city']],
            'runs_left': [runs_left],
            'balls_left': [balls_left],
            'wickets_left': [10 - wickets],
            'total_runs_x': [target],
            'crr': [crr],
            'rrr': [rrr]
        })

        print("📊 T20 Input DF:\n", input_df)

        result = t20_model.predict_proba(input_df)
        return jsonify({
            'win_prob': round(result[0][1] * 100, 2),
            'loss_prob': round(result[0][0] * 100, 2)
        })

    except Exception as e:
        print("🔥 T20 Exception:", e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)