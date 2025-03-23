import os
os.environ["PYVISTA_OFF_SCREEN"] = "true"
os.environ["VTK_USE_OFFSCREEN"] = "1"
os.environ["DISPLAY"] = ""

import tempfile
from flask import Flask, request, jsonify, send_from_directory
from pipeline import run_pipeline

app = Flask(__name__, static_url_path='/NeuronB/public')

@app.route("/analyze", methods=["POST"])
def analyze():
    flair_file = request.files.get("flair")
    t1gd_file  = request.files.get("t1gd")
    prompt     = request.form.get("prompt")

    with tempfile.TemporaryDirectory() as tmpdir:
        flair_path = os.path.join(tmpdir, "flair.nii.gz")
        t1gd_path  = os.path.join(tmpdir, "t1gd.nii.gz")

        flair_file.save(flair_path)
        t1gd_file.save(t1gd_path)

        result = run_pipeline(flair_path, t1gd_path)
        return jsonify(result)
    
@app.route('/renders/<path:filename>')
def serve_render(filename):
    return send_from_directory('renders', filename)

if __name__ == "__main__":
    os.makedirs("renders", exist_ok=True)
    app.run(host="0.0.0.0", port=8000, threaded=False)
