import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image
import pickle
from streamlit_option_menu import option_menu
from streamlit_extras.switch_page_button import switch_page
from tensorflow import keras
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# charger le css
# with open('style.css') as css:
#         st.markdown(f'<style>{css.read}</style>', unsafe_allow_html=True)


# def local_css(file_name):
#     with open(file_name) as c:
#         st.markdown(f'<style>{c.read()}</style>', unsafe_allow_html=True)

# local_css("style.css")

# # Charger le modèle entraîné
# with open('model_3.pickle', 'rb') as f:
#         model = pickle.load(f)


######################################################################################################

# Initialiser la variable de session pour le choix de la page
# if 'page_choice' not in st.session_state:
#     st.session_state.page_choice = 'ACCEUIL'



# Créer la barre de menu avec st.sidebar
# menu = ['ACCEUIL', 'GAME 1', 'GAME 2']
# choice = st.sidebar.selectbox('CHOISISSEZ UNE PAGE', menu, index=menu.index(st.session_state.page_choice))

# the side bar that contains radio buttons for selection of game
# with st.sidebar:
#     game = st.radio('SELECT A GAME',
#     ('ACCEUIL', 'GAME 1', 'GAME 2'),
#     index=('ACCEUIL', 'GAME 1', 'GAME 2').index(st.session_state.page_choice))


# Stocker la valeur de la page sélectionnée dans la variable de session
# st.session_state.page_choice = choice if choice != 'ACCEUIL' else game

# Configurer l'application Streamlit
st.set_page_config(page_title="NBRECO", page_icon=":pencil2:", layout="wide")

# import du modèle
model = keras.models.load_model('modeloo.h5')

model_aug = keras.models.load_model('model_aug.h5')

df = pd.read_csv('data/train.csv')

# séparer les features de la target
X = df.drop(["label"], axis = 1)
y = df["label"]

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

X_test_arr = X_test.values.reshape(-1, 28, 28, 1)

with st.container():

    with st.container():
        selected = option_menu(
            menu_title=None,
            options=["ACCEUIL", "GAME 1", "GAME 2"],
            icons=['house', 'cloud-upload', "graph-up-arrow"],
            menu_icon="cast",
            orientation="horizontal",
            styles={
                "nav-link": {
                    "text-align": "left",
                    "--hover-color": "#ffc107",
                },

                "nav-link-selected": {"background-color": "#ffc107"},


            }
        )

        if selected == "ACCEUIL":
            st.subheader('ACCEUIL')
            st.title('Bienvenue !')
            st.header('Tester notre application et tester les prédictions de notre modèle !')
            st.header("Voici l'architecture du modèle de prédiction :")
            successive_outputs = [layer.output for layer in model.layers[0:]]
            visualization_model = Model(inputs = model.input, outputs = successive_outputs)
            test = ((X_test_arr).reshape((-1,28,28,1)))/255.0
            successive_feature_maps = visualization_model.predict(test)
            layer_names = [layer.name for layer in model.layers]
            for layer_name, feature_map in zip(layer_names, successive_feature_maps):
                if len(feature_map.shape) == 4:
                        n_features = feature_map.shape[-1]
                        size = feature_map.shape[ 1]
                        display_grid = np.zeros((size, size * n_features))
                        for i in range(n_features):
                                x  = feature_map[-1, :, :, i]
                                x -= x.mean()
                                x /= x.std ()
                                x *=  64
                                x += 128
                                x  = np.clip(x, 0, 255).astype('uint8')
                                display_grid[:, i * size : (i + 1) * size] = x
                                scale = 20. / n_features
                        fig = plt.figure( figsize=(scale * n_features, scale) )
                        plt.title ( layer_name )
                        plt.grid  ( False )
                        plt.imshow( display_grid, aspect='auto', cmap='viridis' )
                        st.pyplot(fig)

        if selected == "GAME 1":
            st.title('Number Recognition')

            # Function to preprocess the image
            def preprocess_image(image):
                # Convert the image to grayscale
                image = image.convert('L')
                # Resize the image to the required input shape of the model
                image = image.resize((28, 28))
                # Invert the pixel values
                image = np.invert(image)
                # Reshape the image to a 4D array with a batch size of 1
                image = np.reshape(image, (1, 28, 28, 1))
                # Normalize the pixel values
                image = image / 255.0
                return image

            # Create a file uploader widget
            uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

            if uploaded_file is not None:
                # Display the uploaded image
                image = Image.open(uploaded_file)

                # Resize the image to a width of 300 pixels and proportional height
                width, height = image.size
                new_width = 600
                new_height = 600
                resized_image = image.resize((new_width, new_height))

                st.image(resized_image, caption='Uploaded Image', use_column_width=False)

                # Preprocess the image
                preprocessed_image = preprocess_image(image)

                # Use the model to predict the number in the image
                prediction = model.predict(preprocessed_image)
                predicted_number = np.argmax(prediction)

                # Display the predicted number
                st.header(f"Predicted number is: {predicted_number}")

            else:
                st.write("Please upload an image file")


        if selected == "GAME 2":
            # Game 2
            st.title('Number Recognition')
            canvas_size = 300
            predictions = []
            n_prediction = st.session_state.get('n_prediction', 0)
            score = st.session_state.get('score', 0)
            max_try = 10
            game_over = False
            try_left = st.session_state.get('try_left', 10)


            canvas = st_canvas(
                fill_color="black",
                stroke_width=10,
                stroke_color="white",
                background_color="black",
                height=300,
                width=300,
                drawing_mode="freedraw",
                key="canvas"
            )

            true_number = st.selectbox("Veuillez saisir le chiffre que vous allez dessiner", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            predict_button = st.button('Predict', key=f"predict")

            def pred_model():
                predictions = []
                img_resized = Image.fromarray(canvas.image_data.astype('uint8')).resize((28, 28))

                # Convert the image to grayscale
                img_gray = img_resized.convert('L')

                # Convertir l'image en array numpy
                img_array = np.array(img_gray)

                # Traiter l'image comme nécessaire (ex: la normaliser)
                processed_img_array = img_array / 255.0

                st.image(processed_img_array)
                # Stocker l'image dans une variable
                image = np.expand_dims(processed_img_array, axis=0)

                # Prédire le chiffre en utilisant le modèle
                prediction = model_aug.predict(image)

                st.write(prediction)
                # Ajouter la prédiction à la liste de prédictions
                predictions.append(np.argmax(prediction))

                return predictions

            def test():
                global n_prediction, score, try_left
                predictions = pred_model()
                # Afficher le résultat de la prédiction

                # Incrémenter le compteur de prédictions
                n_prediction += 1
                try_left -= 1

                # Vérifier si la prédiction est correcte
                if np.argmax(predictions) == true_number:
                    score += 1
                    st.header(f"Votre chiffre est : {predictions[0]} !")
                else:
                    st.header(f"Votre chiffre est : {predictions[0]} !")

                # Stocker les nouvelles valeurs dans st.session_state
                st.session_state['n_prediction'] = n_prediction
                st.session_state['score'] = score
                st.session_state['try_left'] = try_left


            ################################################################################

            def play():
                global n_prediction, score, try_left, game_over
                # Prédire le chiffre dessiné par l'utilisateur

                if predict_button:
                    test()

                else :
                    st.write("appuyer sur le bouton predict !")

                if n_prediction == max_try:
                    # Calculer le score final
                    score_ratio = score / max_try

                    # Afficher les statistiques
                    st.write("Le jeu est terminé !")
                    st.write(f"Vous avez fait {max_try} tentatives, et votre score est de {score}/{max_try}.")
                    st.write(f"Votre ratio de bonnes réponses est de {score_ratio:.2f}.")



            play()
