import time
import json
from preprocessing import FileIO
from typing import List
import tiktoken 
from loguru import logger
from prompt_templates import context_block, question_answering_prompt_series
import streamlit as st  
from weaviate_interface import WeaviateClient, WhereFilter

@st.cache_data
def load_content_cache(data_path: str):
    data = FileIO().load_parquet(data_path)
    
    # Convert data into a dictionary for easier access
    data_dict = {d['doc_id']: d for d in data}

    # Function to build a doc_id
    def build_doc_id(video_id, number):
        return f"{video_id}_{number}"

    # Function to get the content of a doc_id if it exists
    def get_content(doc_id):
        return data_dict.get(doc_id, {}).get('content', '')

    # Map each doc_id to a dictionary of all its values with extended content
    content_data = {}
    for d in data:
        doc_id = d['doc_id']
        video_id, number_str = doc_id.rsplit('_', 1)
        number = int(number_str)

        # Get the content of the previous and next doc_ids
        prev_content = get_content(build_doc_id(video_id, number - 1))
        next_content = get_content(build_doc_id(video_id, number + 1))

        # Extend the current content with the previous and next contents
        extended_content = prev_content + d.get('content', '') + next_content

        # Add to the content_data with all values except 'content' and 'content_embedding' 
        content_data[doc_id] = {key: value for key, value in d.items() if key not in ['content', 'content_embedding']}
        content_data[doc_id]['extended_content'] = extended_content

    return content_data

@st.cache_data
def load_guests(data_path: str):
    with open(data_path, 'r') as f:
        data = json.load(f)
    # Remove 'content' key from each dictionary in the list
    return sorted(list(set([d['guest'] for d in data])))

def load_topic_lookup(topic_path: str):
    with open(topic_path, 'r') as f:
        topic_lookup = json.load(f)
    return topic_lookup

def load_topics_list():
    return sorted(set([
                        'Other',
                        'Personal Growth and Self-Care',
                        'Personal Development',
                        'Health and Aging',
                        'Success Mindset',
                        'Impact Investing and Social Responsibility',
                        'Mindset and Personal Growth',
                        'Meditation and Consciousness',
                        'Future of Work and AI',
                        'Economic Outlook and Financial Strategy',
                        'Financial Literacy and Wealth Accumulation',
                        'Success Stories of Celebrities',
                        'Morning Routines for Peak Performance',
                        'Modern Dating Dynamics'
                    ]))

def convert_seconds(seconds: int):
    """
    Converts seconds to a string of format Hours:Minutes:Seconds
    """
    return time.strftime("%H:%M:%S", time.gmtime(seconds))

def generate_prompt_series(query: str, results: List[dict]) -> str:
    """
    Generates a prompt for the OpenAI API by joining the context blocks of the top results.
    Provides context to the LLM by supplying the summary, guest, and retrieved content of each result.

    Args:
    -----
        query : str
            User query
        results : List[dict]
            List of results from the Weaviate client
    """
    context_series = '\n'.join([context_block.format(summary=st.session_state['doc_lookup'][res['doc_id']]['summary'],
                                                    guest=st.session_state['doc_lookup'][res['doc_id']]['guest'], \
                                                    transcript=st.session_state['doc_lookup'][res['doc_id']]['extended_content']) for res in results]).strip()
    prompt = question_answering_prompt_series.format(question=query, series=context_series)
    return prompt

def validate_token_threshold(ranked_results: List[dict], 
                             base_prompt: str,
                             query: str,
                             tokenizer: tiktoken.Encoding, 
                             token_threshold: int, 
                             verbose: bool = False
                             ) -> List[dict]:
        """
        Validates that prompt is below the set token threshold by adding lengths of:
            1. Base prompt
            2. User query
            3. Context material
        If threshold is exceeded, context results are reduced incrementally until the 
        combined prompt tokens are below the threshold. This function does not take into
        account every token passed to the LLM, but it is a good approximation.
        """
        overhead_len = len(tokenizer.encode(base_prompt.format(question=query, series='')))
        context_len = _get_batch_length(ranked_results, tokenizer)
    
        token_count = overhead_len + context_len
        if token_count > token_threshold:
            print('Token count exceeds token count threshold, reducing size of returned results below token threshold')
            
            while token_count > token_threshold and len(ranked_results) > 1:
                num_results = len(ranked_results)
                
                # remove the last ranked (most irrelevant) result
                ranked_results = ranked_results[:num_results-1]
                # recalculate new token_count
                token_count = overhead_len + _get_batch_length(ranked_results, tokenizer)

        if verbose:
            logger.info(f'Total Final Token Count: {token_count}')
        return ranked_results

def _get_batch_length(ranked_results: List[dict], tokenizer: tiktoken.Encoding) -> int:
    '''
    Convenience function to get the length in tokens of a batch of results 
    '''
    contexts = tokenizer.encode_batch([r['content'] for r in ranked_results])
    context_len = sum(list(map(len, contexts)))
    return context_len

MODEL_LOOKUP = {
    'Impact_theory_mpnet_256': {"model": 'models/finetuned-bge-base-en-v1.5-300/', "data": 'processed_data/all-mpnet-base-v2-256.parquet'},
    'Impact_theory_mpnet_finetuned_256': {"model": 'models/finetuned-all-mpnet-base-v2-300/', "data": 'processed_data/finetuned-all-mpnet-base-v2-300-256.parquet'},
    'Impact_theory_bge_256': {"model": 'BAAI/bge-base-en-v1.5', "data": 'processed_data/bge-base-en-v1.5-256.parquet'},
    'Impact_theory_bge_finetuned_256': {"model": 'models/finetuned-bge-base-en-v1.5-300/', "data": 'processed_data/finetuned-bge-base-en-v1.5-256.parquet'},
    'Impact_theory_bge_large_256': {"model": 'BAAI/bge-large-en-v1.5', "data": 'processed_data/bge-large-en-v1.5-256.parquet'},
    'Impact_theory_bge_large_finetuned_256': {"model": 'models/finetuned-bge-large-en-v1.5-300/', "data": 'processed_data/finetuned-bge-large-en-v1.5-256.parquet'},
    'Impact_theory_MiniLM_finetuned_256': {"model": 'models/finetuned-all-MiniLM-L6-v2-300/', "data": 'processed_data/impact-theory-minilmL6-256-2023-12-05_22-48-56.parquet'}
}

def update_index(class_name: str, limit: int, alpha: float, temperature: float, model_lookup: dict= MODEL_LOOKUP):
    st.session_state['class_name'] = class_name
    st.session_state['model_name'] = model_lookup[class_name]['model']
    st.session_state['weaviate_client'] = WeaviateClient(st.session_state['api_key'], st.session_state['url'], model_name_or_path=st.session_state['model_name'])
    st.session_state['doc_lookup'] = load_content_cache(model_lookup[class_name]['data'])
    st.session_state['limit'] = limit
    st.session_state['alpha'] = alpha
    st.session_state['temperature'] = temperature

def run_hybrid_search(client: WeaviateClient, query: str, class_name: str, guest: str, topic: str, limit:int):
    guest_filter = None
    topic_filter = None
    if guest:
        guest_filter = WhereFilter(['guest'], operator='Equal', valueText=guest).todict()
    if topic:
        topic_filter = WhereFilter(['topic'], operator='Equal', valueText=topic).todict()
    
    multi_filter = {"operator": "And","operands": [guest_filter, topic_filter]}
    # make hybrid call to weaviate
    display_props = ['doc_id',
                        'video_id', 
                        'content',
                        ]
    if guest_filter and topic_filter:
        multi_filter = {"operator": "And","operands": [guest_filter, topic_filter]}
        return client.hybrid_search(query, class_name, limit=limit, display_properties=display_props, alpha=st.session_state['alpha'], where_filter=multi_filter)
    elif guest_filter or topic_filter:
        where_filter = guest_filter or topic_filter
        return client.hybrid_search(query, class_name, limit=limit, display_properties=display_props, alpha=st.session_state['alpha'], where_filter=where_filter)
    else:
        return client.hybrid_search(query, class_name, limit=limit, display_properties=display_props, alpha=st.session_state['alpha'])

def search_result(i: int, 
                  url: str, 
                  title: str, 
                  content: str,
                  guest: str,
                  topic: str,
                  length: str,
                  space: str='&nbsp; &nbsp;'
                 ) -> str:
    
    '''
    HTML to display search results.

    Args:
    -----
    i: int
        index of search result
    url: str
        url of YouTube video 
    title: str
        title of episode 
    content: str
        content chunk of episode
    '''
    return f"""
        <div style="font-size:120%;">
            {i + 1}.<a href="{url}">{title}</a>
        </div>

        <div style="font-size:95%;">
            <p>Guest: {guest} {space}{space} Topic: {topic}</p>
            <p>Episode Length: {length}</p>
            <div style="color:grey;float:left;">
                ...
            </div>
            {content}
        </div>
    """