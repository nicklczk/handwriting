from flask import Flask, render_template, jsonify, request, Response, redirect, url_for
import os, sys
import services.net.classify_nn

app = Flask(__name__)

'''
Create a list containing all sub-directories in the input path
'''
def list_subdirs(dir_name):
    dir_list = os.listdir(dir_name)
    dir_list.sort()
    dir_list = [os.path.join(dir_name, name) for name in dir_list if os.path.isdir(os.path.join(dir_name, name))]
    return dir_list

def list_classifiers(dir_name):
    cl_list = os.listdir(dir_name)
    cl_list.sort()
    cl_list = [os.path.join(dir_name, name) for name in cl_list if os.path.splitext(name.lower())[1] in [".pt"]]
    return cl_list

################################################################################
###  Index  ####################################################################
################################################################################
'''
GET /
    Return static page core.html
'''
@app.route("/")
def hello_name():
    return render_template("core.html")


################################################################################
###  Upload  ###################################################################
################################################################################
'''
GET /Upload
    Return: static page upload.html
'''
@app.route("/upload", methods=["GET"])
def upload_file_get():
    return render_template("upload.html")

'''
POST /Upload
    Input: request object...
        API key
        Image file in request.files["img_file"]
    Modifies: Saves an image to /temp_files/upload_<GUID>.ext
    Return: JSON object containing...
        GUID
        Output file name
'''
@app.route("/upload", methods=["POST"])
def upload_file_post():
    # Request data
    r = request

    # Verify API key

    # Check that a file was sent
    if "img_file" not in r.files:
        return "Error: File not posted"

    # Read file data
    file = r.files["img_file"]

    # Is file empty?
    if file.filename == "":
        return "No file selected"

    # Generate unique ID
    file_guid = None

    # Create out file name
    filename = secure_filename(file.filename)

    # Save the file
    file.save(filename)

    # POST return data
    data_out = {
        "file_uid" : "",
        "filename_out" : "",
    }

    return jsonify(data_out)

################################################################################
###  Label Image  ##############################################################
################################################################################
'''
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
'''
@app.route("/label_image", methods=["POST"])
def label_image_post():
    # Request data
    r = request

    # This should hold deserialized JSON input data
    data_in = {}

    # Verify API key

    # Load images from uid

################################################################################
###  Extract Text Area  ########################################################
################################################################################
'''
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
'''
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
    data = {
        **data_in,
        "filename_out" : "",
        "regions" : [],
    }

    return jsonify(data)


################################################################################
###  Extract Lines  ############################################################
################################################################################
'''
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
'''
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
    data = {
        **data_in,
        "filename_out" : "",
        "lines" : []
    }

    return jsonify(data)

################################################################################
###  Extract Letters  ##########################################################
################################################################################
'''
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
'''
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
    data_out = {
        **data_in,
        "filename_out" : "",
        "letters" : []
    }

    return jsonify(data_out)

################################################################################
###  Classifier Statistics  ####################################################
################################################################################
'''
POST /classifier/statistics
    Input: request object...
        API key
        Classifier name
    Return: JSON object...
        Input data
'''
@app.route("/classifier/stats", methods=["POST"])
@app.route("/classifier/statistics", methods=["POST"])
def classifier_statistics_post():
    # Get api_key
    r = request
    api_key = request.form.get("api_key") if "api_key" in request.form else ""
    classifier_name = request.form.get("classifier_name") if "classifier_name" in request.form else ""

    # Grab all subdirectories of ~/classifiers
    cl_path = os.path.join(app.root_path, "classifiers")
    dirs = list_subdirs(cl_path)

    # Grab dir ~/classifiers/<api_key> if it exists
    api_dir = os.path.join(cl_path, api_key) if os.path.join(cl_path, api_key) in dirs else ""

    if (api_dir != ""):
        filepath = os.path.join(api_dir, classifier_name)
        if (os.path.isfile(filepath)):
            data = classify_nn.read_json(filepath)
            return jsonify(data)
        else:
            return jsonify({"error":"No statistics available."})
    else:
        return jsonify({"error":"Api key not recognized."})


@app.route("/classifiers", methods=["GET"])
def classifiers_get():
    return render_template("classifiers.html")


@app.route("/classifiers", methods=["POST"])
def classifiers_post():
    # Get api_key
    r = request
    api_key = request.form.get("api_key") if "api_key" in request.form else ""

    # Grab all subdirectories of ~/classifiers
    cl_path = os.path.join(app.root_path, "classifiers")
    dirs = list_subdirs(cl_path)

    # Grab dir ~/classifiers/<api_key> if it exists
    api_dir = os.path.join(cl_path, api_key) if os.path.join(cl_path, api_key) in dirs else ""

    # if (api_dir != ""):
    #     classify_nn.

    return jsonify({"classifiers":list_classifiers(api_dir)})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
