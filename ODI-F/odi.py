from flask import Flask, request, jsonify
from flask import Flask, render_template
from flask_cors import CORS
import pickle
import pandas as pd

app = Flask(__name__, template_folder="Front-end") 
CORS(app, resources={r"/*": {"origins": "*"}})

# Load ODI model
try:
    odi_model = pickle.load(open(r"ODI-F/odimodel.pkl", "rb"))
except Exception as e:
    print(f"Error loading ODI model: {e}")
    odi_model = None

@app.route('/predict/odi', methods=['POST'])
def predict_odi():
    if not odi_model:
        return jsonify({'error': 'ODI Model not loaded'}), 500

    try:
        data = request.get_json(force=True)
        print("✅ ODI Data Received:", data)

        # ✅ Collect multiple validation errors
        errors = []

        overs = float(data.get('overs', 0))
        wickets = int(data.get('wickets', 0))
        score = int(data.get('score', 0))
        target = int(data.get('target', 0))

        # Validation for ODI (50 overs max)
        if not (0 < overs <= 50):
            errors.append("Overs must be between 0 and 50")
        if not (0 <= wickets < 10):
            errors.append("Wickets must be between 0 and 9")

        if errors:
            return jsonify({'errors': errors}), 400

        runs_left = target - score
        balls_left = 300 - int(overs * 6)  # 50 overs × 6
        crr = score / overs
        rrr = (runs_left * 6 / balls_left) if balls_left > 0 else 0

        # ✅ High-momentum override (optional UX fix)
        if crr > rrr and crr > 12 and overs <= 10:
            return jsonify({
                'win_prob': 80.0,
                'loss_prob': 20.0,
                'note': 'High momentum phase – adjusted prediction'
            })

        # ✅ Use wickets fallen (same as IPL model)
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

        print("📊 ODI Input DF:\n", input_df)

        result = odi_model.predict_proba(input_df)
        return jsonify({
            'win_prob': round(result[0][1] * 100, 2),
            'loss_prob': round(result[0][0] * 100, 2)
        })

    except Exception as e:
        print("🔥 ODI Exception:", e)
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)