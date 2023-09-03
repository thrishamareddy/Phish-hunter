import streamlit as st
import machine_learning as ml
import feature_extraction as fe
from bs4 import BeautifulSoup
import requests as re
import matplotlib.pyplot as plt
st.title('Phish Hunter')
with st.expander("PROJECT DETAILS"):
    st.subheader('Approach')
    st.write('I used supervised learning to classify phishing and legitimate websites. '
             'I benefit from content-based approach and focus on html of the websites. '
             'Also, I used scikit-learn for the ML models.'
             )
    st.write('Features + URL + Label ==> Dataframe')
    st.markdown('label is 1 for phishing, 0 for legitimate')
    number = st.slider("Select row number to display", 0, 100)
    st.dataframe(ml.legitimate_df.head(number))


    @st.cache
    def convert_df(df):
        return df.to_csv().encode('utf-8')
    csv = convert_df(ml.df)
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='phishing_legitimate_structured_data.csv',
        mime='text/csv',
    )

    st.subheader('Results')
    st.write('I used  different ML classifiers of scikit-learn and tested them implementing k-fold cross validation.'
             'Firstly obtained their confusion matrices, then calculated their accuracy, precision and recall scores.'
             'Comparison table is below:')
    st.table(ml.df_results)
    st.write('SVM --> Support Vector Machine')
    st.write('DT --> Decision Tree')
    st.write('RF --> Random Forest')


choice = st.selectbox("Please select your machine learning model",
                 [
                     'Support Vector Machine', 'Decision Tree', 'Random Forest'
                 ]
                )

model = ml.nb_model


if choice == 'Support Vector Machine':
    model = ml.svm_model
    st.write('SVM model is selected!')
elif choice == 'Decision Tree':
    model = ml.dt_model
    st.write('DT model is selected!')
elif choice == 'Random Forest':
    model = ml.rf_model
    st.write('RF model is selected!')


url = st.text_input('Enter the URL')
if st.button('Check!'):
    try:
        response = re.get(url, verify=False, timeout=4)
        if response.status_code != 200:
            print(". HTTP connection was not successful for the URL: ", url)
            st.warning("Attention! This web page is a potential PHISHING!")
        else:
            soup = BeautifulSoup(response.content, "html.parser")
            vector = [fe.create_vector(soup)]
            result = model.predict(vector)
            if result[0] == 0:
                st.success("This web page seems a legitimate!")
                st.balloons()
            else:
                st.warning("Attention! This web page is a potential PHISHING!")
                st.snow()

    except re.exceptions.RequestException as e:
        print("--> ", e)