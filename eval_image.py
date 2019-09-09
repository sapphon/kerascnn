import cv2
import tensorflow as tf

CATEGORIES = range(10)

def prepare(file):
    IMG_SIZE = 28
    img_array = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

model = tf.keras.models.load_model("venv/CNN.model")
image = "/Users/cshaugh1/Downloads/t7.png"
prediction = model.predict(prepare(image))
prediction = list(prediction[0])
sortedPredictions = sorted(prediction)
print("Most likely: " + str(CATEGORIES[prediction.index(sortedPredictions.pop())]))
print("If not that, maybe: " + str(CATEGORIES[prediction.index(sortedPredictions.pop())]))
print("If it's not: " + str(CATEGORIES[prediction.index(sortedPredictions.pop())]) + ", I give up.")