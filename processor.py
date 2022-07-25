import nltk
#nltk.download('punkt')
nltk.download()
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
#import imagepy as imp

#import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
from keras.models import load_model
from keras.preprocessing import image
import json
import random
#global graph
#graph = tf.get_default_graph()

#with graph.as_default():
model = load_model('models/chatbot_model.h5')
image_model = load_model('models/image_model.h5', custom_objects={'KerasLayer':hub.KerasLayer} )
intents = json.loads(open('intents.json', encoding='utf-8').read())
words = pickle.load(open('models/words.pkl','rb'))
classes = pickle.load(open('models/classes.pkl','rb'))



def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
        else:
            result = "You must ask the right questions"
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res
	
def image_response(img_path):
    #dataset_labels = imp.dataset_labels
    dataset_labels = (['Abs System','Airbag Srs','Battery Charge','Brake System',
        'Check Engine','Diesel Particulate Filter', 'Electric Power Steering',
        'High Engine Coolant Temperature' 'Low Fuel' 'Master System Warning',
        'Oil Pressure' 'Tyre Pressure'])
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)  # convert image to numpy arry
    img /= 255
    img = img.reshape((1,) + img.shape)
    #img = np.expand_dims(img, axis=0)
    tf_model_predictions = image_model.predict(img)
  #  dist = np.linalg.norm(img - tf_model_predictions)
    img_pred = dataset_labels[np.argmax(tf_model_predictions)]
    #if dist <= 20:
     #   img_pred = dataset_labels[np.argmax(tf_model_predictions)]
   # else:
   #     img_pred = "This_is_not_the_right_image"
    ints = predict_class(img_pred, model)
    res = getResponse(ints, intents)
    return res
	 
