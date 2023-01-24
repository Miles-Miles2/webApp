
#required assets:

#assets/
#negative.jpg - Image used in about
#positive.jpg - Image used in about
#featureized_data - CSV generated from scripts/featureizer_downloader.py
#serviceAccount - Firebase credentials
#Final_model.h5 - Model generated from src/train.py

import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from firebase import Firebase
import numpy as np
from keras.applications.resnet import preprocess_input
import tensorflow as tf

#configuration
config = {
    "apiKey": "AIzaSyATR_Os7SgthBnMctinbe_eJTZq_X9PxGo",
    "authDomain": "concretecrack-d1fb3.firebaseapp.com",
    "projectId": "concretecrack-d1fb3",
    "storageBucket": "concretecrack-d1fb3.appspot.com",
    "messagingSenderId": "70498922134",
    "appId": "1:70498922134:web:7346e052c168adb6a41cb7",
    "measurementId": "G-G7H10P03B5",
    "serviceAccount": "serviceAccount.json",
    "databaseURL": "https://concretecrack-d1fb3-default-rtdb.firebaseio.com/"
}

#functions
def convert_test_data(img_path):
  
  import numpy as np
  import os
  import tensorflow as tf

  from keras.applications.resnet import preprocess_input
  from keras.preprocessing.image import ImageDataGenerator
  from keras.layers import Dense,GlobalAveragePooling2D
  from keras.models import Model

  from keras.layers import Dense,GlobalAveragePooling2D
  from keras.callbacks import EarlyStopping
  from tensorflow import keras

  IMG_SIZE = (224, 224)
  data = []
  labels = []


  # Download the model, valid alpha values [0.25,0.35,0.5,0.75,1]
  base_model = tf.keras.applications.ResNet50(input_shape=(224, 224 ,3), include_top=False, weights='imagenet')
  # Add average pooling to the base
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  model_frozen = Model(inputs=base_model.input,outputs=x)
  # model_frozen.save("/content/drive/MyDrive/AIClub/featurization_model.h5")
  # print("Model Saved")
  # model_frozen = keras.models.load_model('featurization_model.h5')

  img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
  img_array = tf.keras.preprocessing.image.img_to_array(img)
  img_batch = np.expand_dims(img_array, axis=0)
  img_preprocessed = preprocess_input(img_batch)
  data.append(model_frozen.predict(img_preprocessed))
  print(img_path)
  labels.append(img_path)

  # Make sure dimensions are (num_samples, 1280)
  #data = [array[:,:600] for array in data]
  data = np.squeeze(np.array(data))
  labels = np.reshape(labels, (-1,1))
  return data, img,labels

def visualizer(_distance, _nbors, number,_img_array ):
  nbor_images = [f"assets/{i}.jpg" for i in range(number)]
  fig, axes = plt.subplots(1, len(_nbors)+1, figsize=(10, 5))

  for i in range(len(_nbors)+1):
      ax = axes[i]
      ax.set_axis_off()
      if i == 0:
          ax.imshow(_img_array)
          ax.set_title("Input")
      else:
          image_final = plt.imread(nbor_images[i-1])
          ax.imshow(image_final)
          # we get cosine distance, to convert to similarity we do 1 - cosine_distance
          ax.set_title(f"Sim: {1 - _distance[i-1]:.2f}")
  st.pyplot(fig)

#Functions to be used
def image_resizer(image1, image2):
  #first image
  #second image will be resized bbased on image1 dimension
  img1 = Image.open(image1)
  size1 = img1.size[0]
  size2 = img1.size[1]

  #resizing the second image
  img2 = Image.open(image2)
  img3 = img2.resize((size1, size2))
  return img3

#prediction function
def prediction(modelname, sample_image, IMG_SIZE = (224,224)):

    #labels
    labels = ["Not Cracked","Cracked"]

    try:
        load_model = tf.keras.models.load_model(modelname)

        img = Image.open(sample_image)
        #img.thumbnail(IMG_SIZE)
        img = img.resize((224,224))
        img1 = np.array(img.getdata()).reshape(img.size[0], img.size[1], 3)
        img2 = np.expand_dims(img1, axis = 0)
        img3 = img2.astype(np.float32)
        img4 = preprocess_input(img3)

        prediction = load_model.predict(img4)
        return labels[int(np.argmax(prediction))]

    except Exception as e:
        st.write("ERROR: {}".format(str(e)))



def image_enhancement(input_image, output_image, sigma_minimum = 1.0, sigma_maximum = 10.0, sigma_steps = 10):
    #import argparse
    import sys
    import itk
    from distutils.version import StrictVersion as VS
    if VS(itk.Version.GetITKVersion()) < VS("5.0.0"):
        print("ITK 5.0.0 or newer is required.")
        sys.exit(1)
    # parser = argparse.ArgumentParser(
    #     description="Segment blood vessels with multi-scale Hessian-based measure."
    # )
    # parser.add_argument("input_image", help = "Input the path of an Image")
    # parser.add_argument("output_image", help = "Path where the Image must be saved")
    # parser.add_argument("--sigma_minimum", type=float, default=1.0)
    # parser.add_argument("--sigma_maximum", type=float, default=10.0)
    # parser.add_argument("--number_of_sigma_steps", type=int, default=10)
    # args = parser.parse_args()
    input_image = itk.imread(input_image, itk.F)
    ImageType = type(input_image)
    Dimension = input_image.GetImageDimension()
    HessianPixelType = itk.SymmetricSecondRankTensor[itk.D, Dimension]
    HessianImageType = itk.Image[HessianPixelType, Dimension]
    objectness_filter = itk.HessianToObjectnessMeasureImageFilter[
        HessianImageType, ImageType
    ].New()
    objectness_filter.SetBrightObject(False)
    objectness_filter.SetScaleObjectnessMeasure(False)
    objectness_filter.SetAlpha(0.5)
    objectness_filter.SetBeta(1.0)
    objectness_filter.SetGamma(5.0)
    multi_scale_filter = itk.MultiScaleHessianBasedMeasureImageFilter[
        ImageType, HessianImageType, ImageType
    ].New()
    multi_scale_filter.SetInput(input_image)
    multi_scale_filter.SetHessianToMeasureFilter(objectness_filter)
    multi_scale_filter.SetSigmaStepMethodToLogarithmic()
    multi_scale_filter.SetSigmaMinimum(sigma_minimum)
    multi_scale_filter.SetSigmaMaximum(sigma_maximum)
    multi_scale_filter.SetNumberOfSigmaSteps(sigma_steps)
    OutputPixelType = itk.UC
    OutputImageType = itk.Image[OutputPixelType, Dimension]
    rescale_filter = itk.RescaleIntensityImageFilter[ImageType, OutputImageType].New()
    rescale_filter.SetInput(multi_scale_filter)
    itk.imwrite(rescale_filter.GetOutput(),output_image)




#############################################
#web app design

#setting the title
st.title("Concrete Crack Classifier")

#creating two tabs
tab1, tab2, tab3 = st.tabs(["📋 Introduction", "📊 Predictions", "🎆 Image Enhancer"])

#introduction
with tab1:
    with st.container():
        #can change the subheader as you wish
        st.subheader("About my AMP")
        #description abouth the project
        st.write("My AMP allows users to upload images of concrete that they suspect is not structurally sound. The uploaded image will be uploaded and analyzed by an AI to determine whether or not the concrete is cracked. It will also use featurization to tell the user which parts of the image a cracked.")
        st.write("The app will also suggest to users what changes should be made in order to fix the issue.")

        #can continue for more description
        #################################

        #can chnage the subheader as you wish
        st.subheader("Example Images")
        
        #creating 2 columns
        col1, col2 = st.columns(2)

        with col1:
            st.image("negative.jpg", caption = "Uncracked concrete")
            #explanation of the image
            with st.expander("More Info"):
                #description about the image
                ###########################
                st.write("This concrete does not have any major defects. No further action needs to be taken.")
        with col2:
            st.image(image_resizer("negative.jpg", "positive.jpg"), caption = "Cracked concrete")
            #explanation of the image
            with st.expander("More Info"):
                #description about the image
                ###########################
                st.write("This concrete has a crack in it and needs to be fixed.")

#predictions
with tab2:
    with st.container():
        #setting a subheader
        #can change as preference
        st.subheader("Upload an image")

        #setting file uploader
        #you can change the label name as your preference
        image1 = st.file_uploader(label="Upload an image",accept_multiple_files=False, help="The image you upload will be analyzed and the result will be returned to you")

        if image1:
            #reading the image
            im = Image.open(image1)
            #save the image


            im.save("assets/example.png")

            #showing the image

            column1, column2 = st.columns(2)

            with column1:
              st.subheader("Uploaded Image:")
              st.image(image1)
            with column2:
              st.subheader("Enhanced Image")
              image_enhancement("assets/example.png","assets/enhance.png",sigma_minimum = 0.5, sigma_maximum = 0.5, sigma_steps = 10)
              st.image("assets/enhance.png")
              st.markdown("*Note: This is with default settings. If you want to customize the settings, use the Image Enhancer tab.")


            #file details
            #to get the file information
            file_details = {
                "file name": image1.name,
                "file type": image1.type,
                "file size": image1.size
            }

            #write file details

            #doens't look good
            #st.write(file_details)

            #image predicting
            response = prediction("Final_model.h5", image1)
            st.subheader("This is **{}**".format(response))
            #getting the image fectures and the image array
            data, img_array,_ = convert_test_data("assets/example.png")

            #importing the features csv
            data1 = pd.read_csv("featureized_data.csv")

            if response == "Not Cracked":
              #selecting features
              data_neg = data1[data1["label"] == "Negative"]
              data_neg.reset_index(drop = True, inplace = True)
              features_neg = data_neg.iloc[:,0:-1]

              st.subheader("Neighbor Images for Good concretes")
              no_nbors = st.slider("Choose the Number of Nighbor Images", 1, 5, 2)

              if no_nbors:

                #finding the Neighbors
                nn = NearestNeighbors(n_neighbors=no_nbors, metric = 'cosine')
                nn.fit(features_neg)

                #finding the distabce and neighbors for imported image
                distance, nbors = nn.kneighbors([data])
                distance = distance[0]
                nbors = nbors[0]

                #downloading the images from the firebase
                firebase = Firebase(config)
                #downloading an image to the firebase
                storage = firebase.storage()
                
                #downloading nbors
                for index, nbrs in enumerate(nbors):
                    image_fb = "Images/Negative/" + str(nbrs) + ".jpg"
                    storage.child(image_fb).download(f"assets/{index}.jpg")

                visualizer(distance, nbors, no_nbors, img_array)
            
            else:
              #selecting features
              data_pos = data1[data1["label"] == "Positive"]
              data_pos.reset_index(drop = True, inplace = True)
              features_pos = data_pos.iloc[:,0:-1]

              st.subheader("Neighbor Images for Cracked Concretes")
              no_nbors = st.slider("Choose the Number of Nighbor Images", 1, 5, 2)

              if no_nbors:

                #finding the Neighbors
                nn = NearestNeighbors(n_neighbors=no_nbors, metric = 'cosine')
                nn.fit(features_pos)

                #finding the distabce and neighbors for imported image
                distance, nbors = nn.kneighbors([data])
                distance = distance[0]
                nbors = nbors[0]

                #downloading the images from the firebase
                firebase = Firebase(config)
                #downloading an image to the firebase
                storage = firebase.storage()
                
                #downloading nbors
                for index, nbrs in enumerate(nbors):
                    image_fb = "Images/Positive/" + str(nbrs) + ".jpg"
                    storage.child(image_fb).download(f"assets/{index}.jpg")

                visualizer(distance, nbors, no_nbors, img_array)
with tab3:
    with st.container():
        #setting a subheader
        #can change as preference
        st.subheader("Please Upload an Image to Enhance it")
        #setting file uploader
        #you can change the label name as your preference
        image1 = st.file_uploader(label="Upload an image",accept_multiple_files=False, help="Upload an image to show the cracks")
        if image1:
            #reading the image
            im = Image.open(image1)
            #save the image

            st.subheader("About parameters:")
            st.markdown("**Sigma Minimum:** Make model more sensitive and highlight smaller cracks")
            st.markdown("**Sigma Maximim:** Highlights large cracks more")
            st.markdown("**Sigma Steps:** Enhances effect of Sigma Maximim")

            im.save("assets/example1.png")
            #header saying tune the parameter
            st.subheader("Tune the parameters")
            sigma_min = st.slider("Sigma Minimum", 0.0, 5.0, 4.0, 0.1)
            sigma_max = st.slider("Sigma Maximum", 0.0, 10.0, 5.0, 0.1)
            no_steps = st.slider("Sigma Steps", 1, 10, 5, 1)
            image_enhancement("assets/example1.png","assets/output1.png", sigma_minimum = sigma_min, sigma_maximum = sigma_max, sigma_steps = no_steps)
            #create two columns
            col1, col2 = st.columns(2)
            with col1:
              st.subheader("Original Image")
              st.image("assets/example1.png", caption = "Uploaded Image")
            with col2:
              st.subheader("Enhanced Image")
              st.image("assets/output1.png", caption = "Enhanced Image")
