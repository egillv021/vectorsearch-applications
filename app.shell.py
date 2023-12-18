from tiktoken import get_encoding
from weaviate_interface import WeaviateClient
from prompt_templates import question_answering_prompt_series, question_answering_system
from openai_interface import GPT_Turbo
from app_features import (convert_seconds, generate_prompt_series, search_result,
                          validate_token_threshold, load_content_cache, load_guests, 
                          load_topic_lookup, load_topics_list, run_hybrid_search, update_index, get_models)
from reranker import ReRanker
from loguru import logger 
import streamlit as st
import time
import os

# load environment variables
from dotenv import load_dotenv
load_dotenv('.env', override=True)
 
## PAGE CONFIGURATION
st.set_page_config(page_title="Impact Theory", 
                   page_icon=None, 
                   layout="wide", 
                   initial_sidebar_state="auto", 
                   menu_items=None)
##############
# START CODE #
##############
data_path = './data/impact_theory_data.json'

## RETRIEVER
if 'weaviate_client' not in st.session_state:
    st.session_state['api_key'] = os.environ['WEAVIATE_API_KEY']
    st.session_state['url'] = os.environ['WEAVIATE_ENDPOINT']
    st.session_state['limit'] = 3
    st.session_state['alpha'] = 0.25
    st.session_state['weaviate_client'] = WeaviateClient(st.session_state['api_key'], st.session_state['url'])

## RERANKER
#maybe pick multiple rerankers
if 'reranker' not in st.session_state:
    st.session_state['reranker'] = ReRanker(model_name='cross-encoder/ms-marco-MiniLM-L-6-v2')

## LLM 
if 'open_ai_client' not in st.session_state:
    st.session_state['temperature'] = 0.1
    st.session_state['open_ai_client'] = GPT_Turbo()

## ENCODING
if 'encoding' not in st.session_state:
    st.session_state['encoding'] = get_encoding('cl100k_base') #encoding for gpt-4, gpt-3.5-turbo, text-embedding-ada-002

## INDEX NAME
if 'all_classes' not in st.session_state:
    st.session_state['all_classes'] = sorted(st.session_state['weaviate_client'].show_classes())

## Topcis
# topics for each episode generated using Topic_Modeling_with_Llama2.ipynb
# This was suppose to be my Ace used for filtering and stuff 
# but I built it into my index which you don't have access to so I had to remove the filter
# but you can see it in the episode metadata at least.
if 'topics_lookup' not in st.session_state:
    json_file_path = 'topics.json'
    st.session_state['topic_lookup'] = load_topic_lookup(json_file_path)

if 'doc_lookup' not in st.session_state:
    st.session_state['doc_lookup'] = load_content_cache("content_cache.json")

#creates list of guests for sidebar
if 'guest_list' not in st.session_state:
    st.session_state['guest_list'] = load_guests(data_path)

##############
#  END CODE  #
##############
    
def main():
    with st.sidebar:
        guest = st.selectbox('Select a guest:', options=st.session_state['guest_list'], index=None, placeholder='Select a Guest')
        # I had to remove this filter because I relized I had built it into my index which you do not have access to,
        # and this was suppose to be me trying to be clever.... So now it's just displayed in the episode metadata
        #topic = st.sidebar.selectbox(label='Select a topic:', options=st.session_state['all_topics'], index=None, placeholder="Select a Topic")
        topic = None

    with st.sidebar.expander("Search configurations", expanded=False):
        with st.form("my_form"):
            if "models" not in st.session_state:
                st.session_state['models'] = get_models() #assumes you have a models directory
            model = st.selectbox(label='Pick a model:', options=st.session_state['models'], index=None)
            class_name = st.selectbox(label='Pick an index:', options=st.session_state['all_classes'], index=None)
            alpha = st.slider(label='Value of alpha:', min_value=0.0, max_value=1.0, value=0.25, step=0.05)
            limit = st.slider(label="Number of documents to return:", min_value=1, max_value=5, value=3, step=1)
            temperature = st.slider(label="LLM temperature: ", min_value=0.0, max_value=2.0, value=0.1, step=0.1)
            submitted = st.form_submit_button("Submit")
            if submitted:
                update_index(class_name, limit, alpha, temperature, model)
    col_logo, col_text, _ = st.columns([2,5,2])
    with col_logo:
        st.image('./assets/echo_of_impact.png', width=250)
    with col_text:
        st.write('\n\n\n\n\n')
        st.subheader(f"Ask Echo the question answering assistant anything related to Impact Theory 🎙️")
        st.write('\n')
        query = st.text_input('Enter your question: ')

    col1, _ = st.columns([8,2])
    with col1:
        if 'class_name' not in st.session_state:
            st.error("Please start setting up the search configuration in the sidebar")

        if query:
            with st.chat_message("user"):
                st.write(f'{query}')
            ##############
            # START CODE #
            ##############
            start_time = time.time()
            hybrid_response = run_hybrid_search(client=st.session_state['weaviate_client'], query=query, class_name=st.session_state['class_name'], guest=guest, topic=topic, limit=st.session_state['limit'])
            if len(hybrid_response) == 0:
                st.info("No search results meet this filtering criteria") 
                st.stop()
            # rerank results
            ranked_response = st.session_state['reranker'].rerank(hybrid_response, query)
            end_time = time.time()
            print(f"Time elapsed: {end_time - start_time} seconds")
            # validate token count is below threshold
            valid_response = validate_token_threshold(ranked_response, 
                                                        question_answering_prompt_series, 
                                                        query=query,
                                                        tokenizer=st.session_state['encoding'], # variable from ENCODING
                                                        token_threshold=4000, 
                                                        verbose=True)
            ##############
            #  END CODE  #
            ##############

            use_llm = True
            if use_llm and valid_response:
                # generate LLM prompt
                prompt = generate_prompt_series(query=query, results=valid_response)
                # prep for streaming response
                with st.spinner('Generating Response...'):
                    #creates container for LLM response
                    chat_container, response_box = [], st.empty()
                    
                    # execute chat call to LLM
                                 ##############
                                 # START CODE #
                                 ##############
                    resp = st.session_state['open_ai_client'].get_chat_completion(
                                                                                    prompt=prompt, 
                                                                                    system_message=question_answering_system,
                                                                                    temperature=st.session_state['temperature'], 
                                                     )
                                 ##############
                                 #  END CODE  #
                                 ##############
                    try:
                        #inserts chat stream from LLM
                        with response_box:
                            content = resp
                            if content:
                                st.toast("Answer generated 🎉")
                                chat_container.append(content)
                                result = "".join(chat_container).strip()
                                with st.chat_message("ai", avatar="assets/ai.png"):
                                    st.write(f'{result}')
                    except Exception as e:
                        print(e)
                        # continue
            ##############
            # START CODE #
            ##############
            st.divider()
            st.subheader("Search Results")
            for i, hit in enumerate(valid_response):
                col1, col2 = st.columns([8, 2], gap='large')
                #document_data = st.session_state['doc_lookup'][hit['doc_id']]
                topic = st.session_state['topic_lookup'][hit['video_id']]
                image = hit['thumbnail_url'] # get thumbnail_url
                episode_url = hit['episode_url']# get episode_url
                title = hit['title']# get title
                show_length = hit['length']# get length
                time_string = convert_seconds(show_length) # convert show_length to readable time string
            ##############
            #  END CODE  #
            ##############
                with col1:
                    st.write(search_result(    
                                            i=i, 
                                            url=episode_url,
                                            guest=hit['guest'],
                                            topic=topic,
                                            title=title,
                                            content=hit['content'], 
                                            length=time_string),
                                unsafe_allow_html=True)
                    st.write('\n\n')
                with col2:
                    st.write(f"<a href='{episode_url}'><img src='{image}' width='200'></a>", unsafe_allow_html=True)


if __name__ == '__main__':
    main()