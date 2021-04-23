from flask import Flask, request, jsonify
from fastai.basic_train import load_learner
from fastai.vision import open_image
from flask_cors import CORS, cross_origin
from PIL import Image

app = Flask(__name__)
CORS(app, support_credentials=True)

# load the learner
learn = load_learner(path='.', file='eye_recycle_trained_model.pkl')
classes = learn.data.classes

image = Image.open(img_file)
image_resized = image.resize((512, 384))
image_resized.save('image_resized.jpg')
 
def predict_single(image_resized):
"function to take image and return prediction"
prediction = learn.predict(image_resized)
    
    
    probs_list = prediction[2].numpy()
    return {
        'category': classes[prediction[1].item()],
        'probs': {c: round(float(probs_list[i]), 5) for (i, c) in enumerate(classes)}
    }


# route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    return jsonify(predict_single(request.files['image']))

if __name__ == '__main__':
    app.run()
