import torch
import numpy as np
import polars as pl
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset import convert_rows_to_feat_dict




def produce_query_embeddings_dict(ground_truth_df):
    return {q: df_q['query','q_emb','v_emb', 's_emb'][0] for (q,), df_q in ground_truth_df.group_by('query')}



def compute_learned_scores(model, suggestions_df, query_embedding_dict):
    device = next(model.parameters()).device

    ids = []
    learned_scores = []
    with torch.no_grad():
        for (query,), query_df in tqdm(suggestions_df.group_by('query'), leave=False, desc="Compute learned scores", total=len(np.unique(suggestions_df['query'])), position=0):
            ids.extend(query_df['id'].to_list())
            if len(query_df['page'].drop_nulls()) == 0:
                learned_scores.extend([None]*len(query_df))
                continue

            query_embedding = query_embedding_dict[query]

            scores = model({k:v.to(device) for k,v in convert_rows_to_feat_dict(query_embedding, query_df).items()})['final_score']
            if scores.dim() == 2:
                scores = scores.squeeze(0)
            learned_scores.extend(scores)

    learned_scores_df = pl.DataFrame({
        'id': pl.Series(ids),
        'learned_score': pl.Series(learned_scores)
    }).sort('id')

    learned_scores_df = learned_scores_df.with_columns(
        learned_score = (pl.col('learned_score')-pl.col('learned_score').min())/(pl.col('learned_score').max()-pl.col('learned_score').min())
     )

    if 'learned_score_old' in suggestions_df.columns:
        suggestions_df = suggestions_df.drop('learned_score_old')

    if 'learned_score' in suggestions_df.columns:
        # important! if previous 'learned_score' is not removed, the subsequent join places the new scores in "learned_score_right" corrupting the evaluation setup
        suggestions_df = suggestions_df.rename({'learned_score': 'learned_score_old'})

    return suggestions_df.join(learned_scores_df, on="id")



def compute_recall_at(k, suggestion_df, sorting_key):
    recalls = []
    for _, qdf in suggestion_df.group_by('query'):
        all_relevant = qdf['relevant'].sum()
        if all_relevant == 0: continue
        topk_relevant = qdf.top_k(k, by=sorting_key)['relevant'].sum()
        recalls.append(topk_relevant/all_relevant)
    return np.mean(recalls), recalls


def plot_recalls(k, suggestion_df, sorting_key):
    recall, recalls = compute_recall_at(k, suggestion_df, sorting_key)
    plt.hist(recalls, bins=100, density=True)
    plt.title(f"R@{k}={recall.item()}")
    plt.xlabel(f"R@{k} per query")
    plt.ylabel("Frequency")
    plt.savefig(f"R@{k}_{sorting_key}.png", dpi=300, bbox_inches="tight")
