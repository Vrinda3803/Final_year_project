import os
import json
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__, static_folder='.')

DATA_PATH = 'data/processed/mmasd_openpose/mmasd_openpose_64frames.npz'
OOF_PATH = 'final_optimization_experiments/rc_bigcn_v2_seed42_oof.npz'

CLASS_NAMES = {
    'as': 'Arm Swings',
    'bs': 'Butterfly Stroke',
    'ce': 'Cat-and-Cow',
    'dr': 'Diagonal Reach',
    'mfs': 'Marching in Place',
    'ms': 'Marching Steps',
    'sac': 'Step and Clap',
    'sq': 'Squats',
    'tr': 'Torso Rotation',
    'tw': 'Toe Walking'
}

print("Loading MMASD dataset & RC-BiGCN predictions...")
data = np.load(DATA_PATH, allow_pickle=True)
oof = np.load(OOF_PATH)

X = data['X']
Q = data['Q']
y = data['y']
clips = data['clips']
classes = list(data['classes'])
probs = oof['probs']
y_pred = oof['y_pred']

@app.route('/')
def index():
    return send_from_directory('.', 'demo_dashboard.html')

@app.route('/api/samples', methods=['GET'])
def get_samples():
    """Returns metadata for all available samples or top representative samples."""
    sample_list = []
    # Pick representative samples for each class
    for c_idx, c_code in enumerate(classes):
        correct = np.where((y == c_idx) & (y_pred == c_idx))[0]
        if len(correct) > 0:
            idx = int(correct[np.argmax(probs[correct, c_idx])])
            sample_list.append({
                'sample_id': idx,
                'clip_name': str(clips[idx]),
                'class_code': c_code,
                'class_name': CLASS_NAMES.get(c_code, c_code),
                'confidence': float(probs[idx, c_idx])
            })
    return jsonify(sample_list)

@app.route('/api/predict/<int:sample_id>', methods=['GET'])
def predict_sample(sample_id):
    """Returns skeleton sequence and model predictions for a specific MMASD sample."""
    if sample_id < 0 or sample_id >= len(X):
        return jsonify({'error': 'Invalid sample_id'}), 400

    coords = X[sample_id, :, :, :2].round(4).tolist()
    conf = Q[sample_id].round(3).tolist()
    p = probs[sample_id]
    
    top3_idx = np.argsort(p)[::-1][:3]
    top3 = [
        {
            'class_code': classes[i],
            'class_name': CLASS_NAMES.get(classes[i], classes[i]),
            'probability': float(p[i])
        }
        for i in top3_idx
    ]

    return jsonify({
        'sample_id': sample_id,
        'clip_name': str(clips[sample_id]),
        'ground_truth_code': classes[y[sample_id]],
        'ground_truth_name': CLASS_NAMES.get(classes[y[sample_id]], classes[y[sample_id]]),
        'predicted_code': top3[0]['class_code'],
        'predicted_name': top3[0]['class_name'],
        'confidence': top3[0]['probability'],
        'top3_predictions': top3,
        'mean_joint_reliability': float(np.mean(Q[sample_id])),
        'frames': coords,
        'confidences': conf
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"\n========================================================")
    print(f" RC-BiGCN Demo Server Running at http://127.0.0.1:{port}")
    print(f" Open http://127.0.0.1:{port} in your browser to test!")
    print(f"========================================================\n")
    app.run(host='127.0.0.1', port=port, debug=False)
