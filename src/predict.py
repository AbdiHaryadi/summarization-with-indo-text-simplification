import sys
import torch
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from models import GraphAlgorithm
from utils.features_utils import  load_sim_emb, generate_sim_table, generate_features, prepare_df, prepare_features
from utils.pas_utils import filter_pas, convert_to_PAS_models, convert_to_extracted_PAS, load_srl_model, predict_srl, filter_incomplete_pas
from utils.main_utils import evaluate, accept_input, create_graph, initialize_nlp, initialize_rouge, load_reg_model, maximal_marginal_relevance, natural_language_generation, preprocess, prepare_df_result, pos_tag, read_data, return_config, transform_summary, semantic_graph_modification
from utils.simplification_utils import generate_simplify_corpus_function


tf.random.set_seed(42)

results_path = 'data/results/'

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1,7"


def main():
    config = return_config(sys.argv)
    
    
    batch = 1
    isOneOnly = config['one_pas_only']
    # Load model
    nlp = initialize_nlp()
    srl_model, srl_data = load_srl_model(config)
    w2v, ft = load_sim_emb(config)
    r_metrics = initialize_rouge()

    loaded = False
    idx = 0

    simplify_corpus = None
    if config["use_simplification"]:
        simplify_corpus = generate_simplify_corpus_function()

    while (True):
        try:
            corpus, corpus_title, ref_sum = accept_input()
        except KeyboardInterrupt:
            break

        # Preprocess
        current_corpus = [preprocess(x) for x in corpus]
        if simplify_corpus is not None:
            current_corpus = simplify_corpus(current_corpus)
            
        current_title = corpus_title

        # Pos Tag            
        corpus_pos_tag = [pos_tag(nlp, sent, i) for i, sent in tqdm(enumerate(current_corpus))]
        # SRL
            
        corpus_pas = [predict_srl(doc, srl_data, srl_model, config) for doc in tqdm(current_corpus)]

        ## Filter incomplete PAS
        corpus_pas = [[filter_incomplete_pas(pas,pos_tag_sent) for pas, pos_tag_sent in zip(pas_doc, pos_tag_sent)]for pas_doc, pos_tag_sent in zip(corpus_pas, corpus_pos_tag)]
        
        ## Cleaning when there is no SRL
        print('Cleaning empty SRL...')
        empty_ids = []
        for i, doc in enumerate(corpus_pas):
            for j, srl in enumerate(doc):
                if srl == []:
                    empty_ids.append([i, j])

        for id in reversed(empty_ids):
            i, j = id
            print(id)
            del corpus_pas[i][j]
            del corpus_pos_tag[i][j]
            
         # Choose one pas
        if (isOneOnly):
            corpus_pas = [[filter_pas(pas, pos) for pas, pos in zip(pas_doc, pos_tag_doc)] for pas_doc, pos_tag_doc in zip(corpus_pas, corpus_pos_tag)]

        print('Extracting features...')
        # Convert to PAS Object
        pas_list = [convert_to_PAS_models(pas, pos) for pas, pos in zip(corpus_pas, corpus_pos_tag)]
        ext_pas_list = [convert_to_extracted_PAS(pas,sent) for pas, sent in zip(corpus_pos_tag, pas_list)]
        ext_pas_flatten = [np.concatenate([sent.pas for sent in doc]) for doc in ext_pas_list]


        # Semantic Graph Formation
        sim_table = generate_sim_table(ext_pas_list, ext_pas_flatten, [w2v, ft])
        graph_list = [create_graph(corpus_pas, sim) for corpus_pas, sim in zip(ext_pas_flatten, sim_table)]
        generate_features(ext_pas_list, sim_table, current_title)

        if (not loaded):
            reg = load_reg_model(config['algorithm'])
    
        total_sum = []
        for i, doc in enumerate(ext_pas_list):
            # Predicting
            df = prepare_df([doc], config, 'pred', 0)
            features_min, features_avg, _ = prepare_features(config, 'pred', df)
            if config['algorithm'] == 'min':
                pred = reg.predict(features_min)
            else:
                pred = reg.predict(features_avg)
            
            semantic_graph_modification(graph_list[i], pred)
            # graph-based ranking algorithm
            graph_algorithm = GraphAlgorithm(graph_list[i], threshold=0.0001, dp=0.85, init=1.0, max_iter=100)
            graph_algorithm.run_algorithm()
            num_iter = graph_algorithm.get_num_iter()
            # maximal marginal relevance 100
            summary = maximal_marginal_relevance(15, 2, ext_pas_list[i], ext_pas_flatten[i], graph_list[i], num_iter, sim_table[i])
            summary_paragraph = natural_language_generation(summary, ext_pas_list[i], ext_pas_flatten[i], corpus_pos_tag[i], isOneOnly)

            hyps = transform_summary(summary_paragraph)
            total_sum.append(hyps)
        
        total_ref = ref_sum
        
        print('Ringkasan oleh model')
        print('----------------')
        print(total_sum[0])

        if (total_ref[0] != []):
            print('Ringkasan referensi')
            print('------------------')
            result = evaluate(r_metrics, total_ref, total_sum, 0)
            print(total_ref[0])
            print('Hasil evaluasi')
            print('---------------')
            res = result.select_dtypes(include=np.number)
            print(res)

if __name__ == "__main__":
    main()