#!/usr/bin/env python3
import logging
import pickle
from os import getenv

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_json("./text_and_entities.json")
with open("./models/tdidf_wm.pkl", "rb") as f:
    tfidf_wm = pickle.load(f)
with open("./models/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
load_dotenv()
GCUBE_TOKEN = getenv("GCUBE_TOKEN")

app = Flask(__name__, static_folder=None)
logging.basicConfig(
    format="%(asctime)s [%(funcName)s]:\t%(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    level=logging.INFO,
)


def query_vectorize(query):
    return vectorizer.transform([query])


def ir_retrieve(query):
    logging.info(f"query={query}")
    query = query_vectorize(query)
    logging.info(f"vectorized_query={query}")
    similarity_arr = cosine_similarity(query, tfidf_wm).flatten()
    best_doc_index_arr = np.argpartition(similarity_arr, -10)[-10:]
    logging.info(f"best_doc_index_arr={best_doc_index_arr}")
    ret = []
    for ind in best_doc_index_arr:
        # Return with HTML tags
        ret.append(df.iloc[ind, 0])
    return ret


def wat_entity_relatedness(query: list, ent: int):
    """
    Returns the relatedness for every (q,ent) pair for q in query
    """
    wat_url = "https://wat.d4science.org/wat/relatedness/graph"
    payload = [
        ("gcube-token", GCUBE_TOKEN),
        ("relatedness", "barabasialbert"),
        ("ids", ent),
    ]
    for id in query:
        payload.append(("ids", id))
    response = requests.get(wat_url, params=payload)
    logging.info(f"wat.d4science.org took {response.elapsed.total_seconds()}")
    response = response.json()
    ret = {}
    # WAT API returns N^2(all possible pairs) results
    # Only keep ones with the entity
    for pair in response["pairs"]:
        if pair["src_title"]["wiki_id"] == ent:
            dst_id = pair["dst_title"]["wiki_id"]
        elif pair["dst_title"]["wiki_id"] == ent:
            dst_id = pair["src_title"]["wiki_id"]
        else:
            continue
        ret[str(dst_id)] = pair["relatedness"]
        logging.info("Calculated relatedness for %s %s", query, ent)
    return ret


def wat_wiki_id(wiki_title: str):
    """Give Wikipedia title returns its wiki_id"""
    wat_url = "https://wat.d4science.org/wat/title"
    payload = [("gcube-token", GCUBE_TOKEN), ("title", wiki_title)]
    logging.info("Requesting wiki_id for %s", wiki_title)
    response = requests.get(wat_url, params=payload)
    ret = response.json()["wiki_id"]
    logging.info("Got %s for %s", ret, wiki_title)
    return ret


def wat_entity_linking(text):
    """
    Entity links a given text
    """
    wat_url = "https://wat.d4science.org/wat/tag/tag"
    payload = [
        ("gcube-token", GCUBE_TOKEN),
        ("text", text),
        ("lang", "en"),
        ("tokenizer", "nlp4j"),
        ("debug", 9),
        (
            "method",
            "spotter:includeUserHint=true:includeNamedEntity=true:includeNounPhrase=true,prior:k=50,filter-valid,centroid:rescore=true,topk:k=5,voting:relatedness=lm,ranker:model=0046.model,confidence:model=pruner-wiki.linear",
        ),
    ]
    response = requests.get(wat_url, params=payload)
    return response.json()["annotations"]


def wat_query_to_id(query: str):
    """Gives the wiki_ids associated with a query after linking"""
    logging.info("Requesting entity linking for %s", query)
    response = wat_entity_linking(query)
    ret = []
    for res in response:
        ret.append(res["id"])
    logging.info("Got entity linked query %s", ret)
    return ret


def relevant_indices(query):
    """
    Returns the 5 most relevant document's indices for a given query
    """
    query_tfidf = vectorizer.transform([query])
    similarity_arr = cosine_similarity(query_tfidf, tfidf_wm).flatten()
    best_doc_index_arr = np.argpartition(similarity_arr, -5)[-5:]
    return best_doc_index_arr


def relevant_entities(query):
    """
    Given a query, returns all the entities found in the relevant documents
    """
    doc_ind = relevant_indices(query)
    ret = set()
    for i in doc_ind:
        ret.update(df.iloc[i, -1])
    return list(ret)


def er_retrieve(query):
    doc_ent = relevant_entities(query)
    logging.info(f"Got the list of relevant entities: {doc_ent}")
    doc_wiki_id = []
    for ent in doc_ent:
        doc_wiki_id.append(wat_wiki_id(ent))
    logging.info("Converted the list of entities to list of wiki_ids")
    query_wiki_id = wat_query_to_id(query)
    logging.info("Converted query into list of wiki_ids")
    ret = {}
    for ind, ent_id in enumerate(doc_wiki_id):
        # query_wiki_id will never be -1. ent_id may be as the links are old
        # Don't waste time calculating for -1
        if ent_id == -1:
            continue
        # WAT API doesn't return anything if the ids are the same
        if ent_id in query_wiki_id:
            ret[doc_ent[ind]] = 1
            continue
        relatedness = list(wat_entity_relatedness(query_wiki_id, ent_id).values())
        if relatedness:
            relatedness = np.average(relatedness)
            ret[doc_ent[ind]] = relatedness
    return ret


@app.route("/erRetrieve/", methods=["GET"])
def handle_er_retrieve():
    query = request.args.get("q")
    return jsonify(er_retrieve(query))


@app.route("/irRetrieve/", methods=["GET"])
def handle_ir_retrieve():
    query = request.args.get("q")
    return jsonify(ir_retrieve(query))


@app.route("/vectorize/", methods=["POST"])
def handle_vectorize():
    query = request.get_json()["query"]
    return jsonify(str(query_vectorize(query)))
