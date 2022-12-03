
import pandas as pd

import streamlit as st
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

seoul_list = pd.read_csv('HotelListInSeoul__en2019100120191005.csv')
seoul_rev = pd.read_csv('hotelReviewsInSeoul__en2019100120191005.csv')

combined = pd.merge(seoul_rev, seoul_list, how="left", left_on=["hotelUrl"], right_on=["url"])


df_combined = combined.sort_values(['hotel_name']).groupby('hotel_name', sort=False).review_body.apply(''.join).reset_index(name='all_review')

import re

df_combined['all_review'] = df_combined['all_review'].apply(lambda x: re.sub('[^a-zA-z0-9\s]','',x))

def lower_case(input_str):
    input_str = input_str.lower()
    return input_str

df_combined['all_review']= df_combined['all_review'].apply(lambda x: lower_case(x))

df_sentences = df_combined.set_index("all_review")
df_sentences = df_sentences["hotel_name"].to_dict()
df_sentences_list = list(df_sentences.keys())
len(df_sentences_list)

embedder = SentenceTransformer('all-MiniLM-L6-v2')

df_sentences_list = [str(d) for d in tqdm(df_sentences_list)]
corpus = df_sentences_list
corpus_embeddings = embedder.encode(corpus,show_progress_bar=True)

df_combined_1 = pd.merge(df_combined, seoul_list, how="left", left_on=["hotel_name"], right_on=["hotel_name"])

import torch

queries = st.text_input("Seoul Hotel Search :")

top_k = min(3, len(corpus))
query_embedding = embedder.encode(queries, convert_to_tensor=True)

cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
top_results = torch.topk(cos_scores, k=top_k)

if queries != '':
    st.write("Search:", "'"+queries+"'")
    st.write("\nHotel Recommendations:", "\n")
    
    for score, idx in zip(top_results[0], top_results[1]):
      row_dict = df_combined_1.loc[df_combined_1['all_review']== corpus[idx]]
    
      st.write(row_dict['hotel_name'].to_string(index=False))
      st.write("(Score: {:.4f})".format(score))
    
      st.write("Hotel Summary:")
      st.write("  City:",row_dict['locality'].to_string(index=False))
      st.write("  Rating:",row_dict['tripadvisor_rating'].to_string(index=False))
      st.write("  Price/Night:",row_dict['price_per_night'].to_string(index=False))
      st.write("  URL:",row_dict['url'].to_string(index=False), "\n")

#!pip freeze > requirements.txt

