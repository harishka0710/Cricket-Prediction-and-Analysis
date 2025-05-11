from flask import Flask, request, jsonify
from flask import Flask, render_template
from flask_cors import CORS
import pickle
import pandas as pd

app = Flask(__name__, template_folder="Front-end") 
CORS(app, resources={r"/*": {"origins": "*"}})


# Load IPL model
try:
    ipl_model = pickle.load(open(r"IPL/pipe.pkl", "rb"))
except Exception as e:
    print(f"Error loading IPL model: {e}")
    ipl_model = None

# Load ODI model
try:
    odi_model = pickle.load(open(r"ODI-F/odimodel.pkl", "rb"))
except Exception as e:
    print(f"Error loading ODI model: {e}")
    odi_model = None

# Load T20 model
try:
    t20_model = pickle.load(open(r"T20/t20model.pkl", "rb"))
except Exception as e:
    print(f"Error loading T20 model: {e}")
    t20_model = None  # Corrected the variable name

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

        if not (0 < overs <= 19):
            errors.append("Overs must be between 0 and 20")
        if not (0 <= wickets < 9):
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
        if not (0 < overs <= 49):
            errors.append("Overs must be between 0 and 50")
        if not (0 <= wickets < 9):
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
        import traceback
        traceback.print_exc()  # 👈 shows full error stack trace in terminal
        return jsonify({'error': str(e)}), 500


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/t20')
def t20():
    return render_template('t20.html')

@app.route('/odi')
def odi():
    return render_template('odi.html')

@app.route('/ipl')
def ipl():
    return render_template('ipl.html')

if __name__ == '__main__':
    app.run(debug=True)
