from flask import (
    Flask,
    render_template,
    jsonify,
    request,
    Response,
    redirect,
    url_for,
    send_from_directory,
)
from werkzeug.utils import secure_filename
import os, sys
from services.net import classify_nn, extract1
import uuid
import cv2

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(app.root_path, "images") # Must specify your upload folder right now
ALLOWED_EXTENSIONS = set(["png", "jpg"])
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

"""
Create a list containing all sub-directories in the input path
"""


def list_subdirs(dir_name):
    dir_list = os.listdir(dir_name)
    dir_list.sort()
    dir_list = [
        os.path.join(dir_name, name)
        for name in dir_list
        if os.path.isdir(os.path.join(dir_name, name))
    ]
    return dir_list


def list_classifiers(dir_name):
    if not os.path.isdir(dir_name):
        return [],[],[]

    cl_files = os.listdir(dir_name)
    cl_files.sort()
    cl_names = [
        os.path.splitext(name)[0]
        for name in cl_files
        if os.path.splitext(name.lower())[1] in [".pt"]
    ]

    stats_files = []
    for cl in cl_names:
        filename = cl+"_stats.json"
        jsn = None
        if (os.path.isfile(filename)):
            jsn = classify_nn.read_json(filename)
        else:
            jsn = "None"

        stats_files.append(os.path.splitext(jsn)[0])

    return cl_names, cl_files, stats_files


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


################################################################################
###  Index  ####################################################################
################################################################################
"""
GET /
    Return static page core.html
"""


@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        # check if the post request has the file part
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        # if user does not select file
        if file.filename == "":
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            return redirect(url_for("uploaded_file", filename=filename))
    return render_template("core.html")


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


################################################################################
###  Upload  ###################################################################
################################################################################
"""
GET /Upload
    Return: static page upload.html
"""
# @app.route('/upload')
# def upload_file():
#    return render_template('upload.html')

# @app.route('/uploader', methods = ['GET', 'POST'])
# def upload_file():
#    if request.method == 'POST':
#       f = request.files['file']
#       f.save(secure_filename(f.filename))
#       return 'file uploaded successfully'

# # POST return data
# data_out = {
#     "file_uid" : "",
#     "filename_out" : "",
# }

# return jsonify(data_out)

################################################################################
###  Label Image  ##############################################################
################################################################################
"""
This is the bread and butter of the application.

POST /label_image
    Input: request object...
        API key
        File file_guid
        Input file __name__
    Modifies: Saves an imae to /temp_files/labeled_<GUID>.ext
    Return: JSON object
        Input data
        Output file __name__
        Line rects
        Line
"""
@app.route("/label_image/<filename>", methods=["GET"])
def label_image_post(filename):

    # Parse request
    api_key = request.form.get("api_key") if "api_key" in request.form else ""
    classifier_name = (
        request.form.get("file_uuid") if "file_uuid" in request.form else ""
    )
    image_name = request.form.get("img_name") if "img_name" in request.form else ""

    class_path = os.path.join("classifiers", "dev_demo_1", "mnist.pt")
    image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    img = cv2.imread(image_path)

    letter_rects = extract1.extract_letters(img)

    predictions = classify_nn.classify(class_path, image_path, letter_rects)

    print(predictions)

    data = {
        "letter_rects" : letter_rects,
        "predictions" : predictions
    }

    return send_from_directory(app.config["UPLOAD_FOLDER"], "out.jpg")

    # # Api key gives the filepath
    # if (api_dir != ""):
    #     filepath = os.path.join(api_dir, classifier_name + ".json")
    #     if (os.path.isfile(filepath)):
    #         data = classify_nn.read_json(filepath)
    #         return jsonify(data)
    #     else:
    #         return jsonify({"error": "No statistics available."})
    # else:
    #     return jsonify({"error": "Api key not recognized."})


################################################################################
###  Extract Text Area  ########################################################
################################################################################
"""
POST /extract_text_area
    Input: request object...
        API key
        File GUID
        Input file name
    Modifies: Saves an image to /temp_files/regions_<GUID>.ext
    Return: JSON object...
        Input data
        Output file name
        Region rects
"""


@app.route("/extract_text_areas", methods=["POST"])
def extract_text_area_post():
    # Request data
    r = request

    # This should hold deserialized JSON input data
    data_in = {}

    # Verify API key

    # Load images from uid

    # Get regions
    # Extractor from our Service package

    # POST return data
    data = {**data_in, "filename_out": "", "regions": []}

    return jsonify(data)


################################################################################
###  Extract Lines  ############################################################
################################################################################
"""
POST /extract_lines
    Input: request object...
        API key
        File GUID
        Input file name
        Region rects
    Modifies: Saves an image to /temp_files/lines_<GUID>.ext
    Return: JSON object...
        Input data
        Output file name
        Line rects
"""


@app.route("/extract_lines", methods=["POST"])
def extract_lines_post():
    # Request data
    r = request

    # This should hold deserialized JSON input data
    data_in = {}

    # Verify API key

    # Load images from file UID

    # Get regions
    # Extractor from our Service package

    # POST return data
    data = {**data_in, "filename_out": "", "lines": []}

    return jsonify(data)


################################################################################
###  Extract Letters  ##########################################################
################################################################################
"""
POST /extract_letters
    Input: request object...
        API key
        File GUID
        Line regions
        Input file name
    Modifies: Saves an image to /temp_files/regions_<GUID>.ext
    Return: JSON object...
        Input data
        Output file name
        Letter rects
"""


@app.route("/extract_letters", methods=["POST"])
def extract_letters_post():
    # Request data
    r = request

    # This should hold deserialized JSON input data
    data_in = {}

    # Verify API key

    # Load images from file UID

    # Get regions
    # Extractor from our Service package

    # POST return data
    data_out = {**data_in, "filename_out": "", "letters": []}

    return jsonify(data_out)


################################################################################
###  Classifier Statistics  ####################################################
################################################################################
"""
POST /classifier/statistics
    Input: request object...
        API key
        Classifier name
    Return: JSON object...
        Input data
"""


@app.route("/classifier/stats", methods=["POST"])
@app.route("/classifier/statistics", methods=["POST"])
def classifier_statistics_post():
    # Get api_key
    r = request
    api_key = request.form.get("api_key") if "api_key" in request.form else ""
    classifier_name = (
        request.form.get("classifier_name") if "classifier_name" in request.form else ""
    )

    # Grab all subdirectories of ~/classifiers
    cl_path = os.path.join(app.root_path, "classifiers")
    dirs = list_subdirs(cl_path)

    # Grab dir ~/classifiers/<api_key> if it exists
    api_dir = (
        os.path.join(cl_path, api_key) if os.path.join(cl_path, api_key) in dirs else ""
    )

    if (api_dir != ""):
        filepath = os.path.join(api_dir, classifier_name + ".json")
        if (os.path.isfile(filepath)):
            data = classify_nn.read_json(filepath)
            return jsonify(data)
        else:
            return jsonify({"error": "No statistics available."})
    else:
        return jsonify({"error": "Api key not recognized."})


@app.route("/classifiers", methods=["GET"])
def classifiers_get():
    return render_template("classifiers.html")


@app.route("/classifiers", methods=["POST"])
def classifiers_post():
    r = request

    # Get api_key
    api_key = request.form.get("api_key") if "api_key" in request.form else ""

    # Grab all subdirectories of ~/classifiers
    cl_path = os.path.join(app.root_path, "classifiers")
    dirs = list_subdirs(cl_path)

    # Grab dir ~/classifiers/<api_key> if it exists
    api_dir = (
        os.path.join(cl_path, api_key) if os.path.join(cl_path, api_key) in dirs else ""
    )

    # if (api_dir != ""):
    #     classify_nn.

    classifiers, cl_files, stats_files = list_classifiers(api_dir)

    data = {}
    for i in range(len(classifiers)):
        data[classifiers[i]] = {
            "model" : cl_files[i],
            "stats" : stats_files[i]
        }

    return jsonify({"classifiers": data})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
