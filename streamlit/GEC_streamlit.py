import streamlit as st
import pandas as pd
import numpy as np
import evaluate
from clova_api import CompletionExecutor
from nltk.translate.bleu_score import sentence_bleu
from clova_api import run_gleu




#Side layout
page = st.sidebar.selectbox("Select a page",["Main Page", "GEC test"])

if page == "Main Page":
    # Display details of Main page
    st.title('Grammatical Error Correction by Hyper Clova🍀')

    st.subheader('GEC : Grammatical Error Correction, 문장 교정')


    #data load
    st.write('Kor-lang8을 hyper clova를 이용해 GEC한 결과입니다')
    st.text('Column : kor-lang8 원문, kor-lang8 교정문, clova교정문, paper GLEU, clova GLEU')
    gleu = run_gleu(reference='result_data_GEC_text.txt', source='result_data_GEC_Completion.txt', hypothesis='result_data_GEC_Correction.txt')
    st.subheader(f'Kor-lang8 GLEU :{gleu}')
    data = pd.read_csv('/VOLUME/grlee/streamlit/result_data_GEC.csv', index_col = 0)
    st.table(data.head(10))

    st.subheader('Sample에 대한 clova의 GEC 성능을 확인할 수 있습니다')
    sample = list(data.Text)
    my_choice = st.selectbox('Select sample sentence !', sample)

    st.write(data[data['Text']==my_choice].Correction)
    st.write(f"GLEU is {str(data[data['Text']==my_choice].GLEU_clova)}")
    


elif page == "GEC test":
    # Display details of Model page
    # model usage 
    st.title('Hyper clova GEC model🍀')
    st.write('Hyper clova를 이용해 문장을 교정할 수 있습니다')

    #model
    completion_executor = CompletionExecutor(
        host='clovastudio.apigw.ntruss.com',
        api_key='NTA0MjU2MWZlZTcxNDJiY0w29xOB9/3nuo/3lerzMMOh5QO1YQGDzUhyeOmA0JO+Vb7WYSTpPrwnInq8VbpyN7gqca9BW0X5yZbqSdLE/2wbq1mXJvBujXcFzfT2OUqFbec6r8A4wCH6mPnhoyOL4B9nn5UciZVJAV9kZEwa5WUPbsNjvmnyt+HhLjXwFXz4T1wpQ5yTc5UqPhDopmfiwg==',
        api_key_primary_val = 'JTmHGtsP1J72wSQq3gltgsizSkKT5AsS6ja0oINt',
        request_id='bd96d59a2d354a899e6a51ec0b1c8a77'
    )

    request_data = {
        'maxTokens': 300,
        'temperature': 0.85,
        'topK': 4,
        'topP': 0.8,
        'repeatPenalty': 5.0,
        'start': '',
        'restart': '',
        'stopBefore': ['<|endoftext|>'],
        'includeTokens': False,
        'includeAiFilters': True,
        'includeProbs': False
    }
    request_data['text'] = st.text_input(label="Sentence", value="교정할 문장을 입력하세요!")
    output_sentence = completion_executor.execute(request_data)


    if st.button("Grammatical Error Correction"):
        con = st.container()
        con.caption("Result")
        con.write(str(output_sentence))
        google_bleu = evaluate.load("google_bleu")
        result = google_bleu.compute(predictions=[request_data['text']], references=[output_sentence])

        con.write(f"GLEU is {str(result['google_bleu'])}")