from typing import Union
from fastapi import FastAPI, UploadFile, File
from typing import Annotated
from fastapi.responses import FileResponse
from fastapi.responses import Response
import uuid
from tempfile import NamedTemporaryFile
from app.ML.predict import make_prediction
import os
from fastapi.middleware.cors import CORSMiddleware




app = FastAPI()
origins = [
    "http://127.0.0.1:5500",  # Your frontend URL
    "http://localhost:5500", 
    "http://127.0.0.1:5500/public/index02.html"   # Additional URLs if needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# @app.post("/files/")
# async def create_file(file: Annotated[bytes, File()]):
#     return {"file_size": len(file)}

# @app.post("/uploadfile/")
# async def create_upload_file(file: UploadFile):
#     return {"filename": file.filename}

# Create a variable to store the uploaded file data
uploaded_file_data = None
@app.get("/")
def new():
    return {"Hello": "World"}

# @app.post("/images/")
# async def create_upload_file(file: UploadFile = File(...)):
#     global uploaded_file_data

#     file.filename = f"{uuid.uuid4()}.jpg"
#     contents = await file.read()

#     # Store the file data in the variable
#     uploaded_file_data = contents

#     return {"filename": file.filename}

# @app.get("/images/")
# async def read_random_file():
#     global uploaded_file_data

#     # Check if there's any uploaded file data
#     if uploaded_file_data is not None:
#         response = Response(content=uploaded_file_data)
#         return response
#     else:
#         return {"message": "No file uploaded yet"}

#Ml Model




# Load and preprocess the image
#file_path = os.path.join(os.path.dirname(__file__), '1420153030.dcm')

#convert to dicom coming soon
#convert_to_dicom('/content/img2.jpg', '/content/Save.dcm', '/content/Save.dcm')
# file_path = 'temp'
# final_img = process(file_path, crop_image=True, size=(TARGET_WIDTH, TARGET_HEIGHT), debug=False, save=False)
@app.post("/images/")
async def create_upload_file(file: UploadFile = File(...)):
    # Generate a unique filename
    #file_path = f"{uuid.uuid4()}.dcm"
    directory = "uploads"
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, f"{uuid.uuid4()}.dcm")
    print('upload path is'+file_path)
    # Use a temporary file for processing
    with open(file_path, "wb") as f:
        contents = await file.read()
        f.write(contents)

        # Process the file asynchronously
        try:
            prediction_value, cancer, accuracy = make_prediction(file_name=file_path, rm_file=True)
        except Exception as e:
            print(f"Error processing image: {e}")
            prediction_value = None
            cancer = None
            accuracy = None
    return {
        "filepath": file_path,
        "prediction_value": prediction_value,
        "cancer": cancer,
        "accuracy": accuracy
    }
# Save the processed image
#output_path = '/content/save/final.jpg'
#cv2.imwrite(output_path, final_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
#print(f"Image saved to {output_path}")

# Display the image

# Expand dimensions to match the expected input shape for the model (1xHxWx1)


# Make a prediction on the processed image



# Example usage
# print(f"Prediction value: {prediction} ")
# accuracy = prediction_accuracy(prediction)
# print(f"Accuracy: {accuracy:.2f}%")
# print(f"Prediction: {np.int8(prediction > THRESHOLD_BEST)} ")
# print(type(prediction))




#Remainging Code


# Set the directory for templates
#templates = Jinja2Templates(directory='templates')

# @app.get("/",)
# def get_login():
#     #return templates.TemplateResponse("index01.html", {"request": request})
#     return {"hello": "world"}
# @app.get("/items")
# def read_item():
#     # Return the prediction value as part of the JSON response
#     return {
#         "prediction_value": prediction_value,  # Send the prediction value
#         "prediction_binary": int(prediction_value > THRESHOLD_BEST),  # Send binary classification
#         "accuracy": round(prediction_value * 100, 2),  # Send the accuracy in percentage
        
#     }

