import torch.nn.functional as F
import torch
import numpy as np
import polars as pl
import editdistance
import evaluate
from torch.utils.data import Dataset

from map_metric import rerank_by, compute_relevance

def load_iam_ground_truth(path_to_gt_parquet):
    return pl.read_parquet(path_to_gt_parquet)

def load_ranked_lists(path_to_kws_output, ground_truth_df, subset, iou_threshold=0.25):
    df = pl.read_parquet(path_to_kws_output)
    df = df.rename({
        'syntactic_embedding': 'v_emb',
        'syntactic_embedding_3levels': 'v3_emb',
        'semantic_embedding': 's_emb',
        "syntactic_similarity": "v_sim",
        "semantic_similarity_using_'all-MiniLM-L12-v2'_trocr_decoded": "s_sim",
        'entropy_trocr': 'ocr_entropy',
    })
    df = df.with_columns( c_emb = pl.concat_arr('v3_emb','s_emb') )
    df = df.rename({'index': 'id'})
    df = rerank_by(df, "v_sim").rename({'ranking_v_sim': 'rank_prior'})
    df = df.with_columns(
        rank_prior = (
            (pl.col("rank_prior") - pl.col("rank_prior").min()) /
            (pl.col("rank_prior").max() - pl.col("rank_prior").min() + 10**-8)
        ).over("query"))
    df = compute_relevance(df, ground_truth_df, subset, iou_threshold)
    return df


class AbstractTrainingSet(Dataset):
    def __init__(self, df, query_embeddings, min_positives_per_query, neg_ratio=10, hard_frac=0.7):
        """
            min_positives_per_query:    1 if PairwiseDataset (positive), 2 if TripletDataset(anchor, positive)
            neg_ratio:                  produces negatives per positive
            hard_frac:                  percentage of hard negatives in negative sample
        """
        super().__init__()

        self.df = df.drop_nulls().sort('id')  # a null value essentially means that there is no prediction for that query; thus it is irrelevant in the reranking
        self.neg_ratio = neg_ratio
        self.hard_frac = hard_frac

        # keep useful queries
        self.queries = []

        keep_ids = []
        for (q,),v in self.df.group_by("query"):
            rel =  v.filter(relevant=True)
            irel = v.filter(relevant=False)

            # no relevant or irrelevant instances in ranked list
            if len(rel) == 0 or len(irel) == 0: continue

            # not enough positive or anchor in ranked list
            if len(rel) < min_positives_per_query: continue

            keep_ids.extend(v['id'].to_list())
            self.queries.append(q)

        self.df = self.df[keep_ids].with_row_index('train_id')

        self.pos_examples = {q: v.filter(relevant=True) for (q,),v in self.df.group_by("query")}
        self.neg_examples = {q: v.filter(relevant=False) for (q,),v in self.df.group_by("query")}

        self.hard_neg_pairs = dict()

        self.query_embeddings = query_embeddings


        self.feat_keys = convert_rows_to_feat_dict(query_embeddings[self.queries[0]], self.df[0]).keys()

    def __len__(self):
        return len(self.queries)

    def mine_hard_negatives(self, compute_learned_scores_fn, margin=0.0):
        df = compute_learned_scores_fn(self.df)['train_id', 'query', 'relevant', 'learned_score']
        pos = df.filter(relevant = True)
        neg = df.filter(relevant = False)
        pairs = (
            pos.join(neg, on="query", suffix="_neg")
                .filter(pl.col('learned_score') <= pl.col("learned_score_neg") + margin)
                .select([
                        "query",
                        pl.col("train_id").alias("pos_id"),
                        pl.col("train_id_neg").alias("neg_id"),
                    ])
        )

        self.df = self.df.sort('train_id')
        self.hard_neg_pairs = { q:qdf for (q,),qdf in pairs.group_by('query') }

    def get_random_negatives(self, query_emb, num):
        query = query_emb['query'].item()
        pos_rows = self.pos_examples[query].sample(n=num, with_replacement=True)
        pos_feats = convert_rows_to_feat_dict(query_emb, pos_rows)
        neg_rows = self.neg_examples[query].sample(n=num, with_replacement=True)
        neg_feats = convert_rows_to_feat_dict(query_emb, neg_rows)
        return pos_feats, neg_feats

    def get_hard_negatives(self, query_emb, num):
        query = query_emb['query'].item()
        if query not in self.hard_neg_pairs:
            return self.get_random_negatives(query_emb, num)

        hard_neg_pool = self.hard_neg_pairs[query].sample(n=num, with_replacement=True)
        pos_rows = self.df[hard_neg_pool['pos_id'].to_list()]
        pos_feats = convert_rows_to_feat_dict(query_emb, pos_rows)
        neg_rows = self.df[hard_neg_pool['neg_id'].to_list()]
        neg_feats = convert_rows_to_feat_dict(query_emb, neg_rows)

        return pos_feats, neg_feats

    def __getitem__(self, idx):
        raise NotImplementedError('this is supposed to be abstract')

    def pick_negatives(self, query, query_emb):
        feats_neg = {k: [] for k in self.feat_keys}

        n_hard = int(np.ceil(self.neg_ratio * self.hard_frac))
        n_rand = self.neg_ratio - n_hard

        # ----- HARD NEGATIVES -----
        if query in self.neg_examples:
            df_hn = self.neg_examples[query]
            hard_neg_pool = df_hn.sample(n=n_hard, with_replacement=True)

            hard_neg_feat = convert_rows_to_feat_dict(query_emb, hard_neg_pool)
            for k,v in hard_neg_feat.items():
                feats_neg[k].append(v)

            # in-query nonhard negatives
            nonhard_neg_pool = self.neg_examples[query].sample(n=n_rand, with_replacement=True)

        else:
            # TODO do we need out-of-query negatives?
            # currently they're only used when a ranked list has a single positvie example

            n_rand = self.neg_ratio

            if n_rand > 0:
                nonhard_neg_pool = []
                while len(nonhard_neg_pool) < n_rand:
                    neg_q = np.random.choice(self.queries)
                    neg_qdf = self.pos_examples[neg_q]
                    if neg_q == query or len(neg_qdf) == 0: continue
                    nonhard_neg_pool.append(neg_qdf.sample(n=1))
                nonhard_neg_pool = pl.concat(nonhard_neg_pool)

        nonhard_neg_feat = convert_rows_to_feat_dict(query_emb, nonhard_neg_pool)
        for k,v in nonhard_neg_feat.items():
            feats_neg[k].append(v)

        return {
            k: torch.cat(v, dim=0)
            for k, v in feats_neg.items()
        }


class PairwiseDataset(AbstractTrainingSet):
    def __init__(self, df, ground_truth, neg_ratio=10, hard_frac=0.7):
        super().__init__(df, ground_truth, min_positives_per_query=1, neg_ratio=neg_ratio, hard_frac=hard_frac)

    def __getitem__(self, idx):
        query = self.queries[idx]
        query_emb = self.query_embeddings[query]

        feats_pos = {k: [] for k in self.feat_keys}
        feats_neg = {k: [] for k in self.feat_keys}

        n_hard = int(np.ceil(self.neg_ratio * self.hard_frac))
        n_rand = self.neg_ratio - n_hard

        pos, neg = self.get_hard_negatives(query_emb, n_hard)
        for k in self.feat_keys:
            feats_pos[k].append(pos[k])
            feats_neg[k].append(neg[k])

        pos, neg = self.get_random_negatives(query_emb, n_rand)
        for k in self.feat_keys:
            feats_pos[k].append(pos[k])
            feats_neg[k].append(neg[k])

        feats_pos = { k: torch.cat(v, dim=0)  for k, v in feats_pos.items() }
        feats_neg = { k: torch.cat(v, dim=0)  for k, v in feats_neg.items() }

        for k in self.feat_keys:
            assert len(feats_pos[k]) == self.neg_ratio
            assert len(feats_neg[k]) == self.neg_ratio

        return feats_pos, feats_neg




class TripletDataset(AbstractTrainingSet):
    def __init__(self, df, ground_truth, neg_ratio=10, hard_frac=0.7):
        super().__init__(df, ground_truth, min_positives_per_query=2, neg_ratio=neg_ratio, hard_frac=hard_frac)

    def __getitem__(self, idx):
        query = self.queries[idx]
        query_emb = self.query_embeddings[query]

        pos_rows = self.pos_examples[query].sample(n=2)
        anc_example = pos_rows[0]
        pos_example = pos_rows[1]

        feats_anc = convert_rows_to_feat_dict(query_emb, anc_example)
        feats_pos = convert_rows_to_feat_dict(query_emb, pos_example)

        feats_neg = self.pick_negatives(query, query_emb)

        for k,v in feats_neg.items():
            assert len(v) == self.neg_ratio

        return feats_anc, feats_pos, feats_neg


def convert_rows_to_feat_dict(query_embedding, rows):
    query = query_embedding['query'].item()
    q_emb = query_embedding['q_emb'].to_torch().to(torch.float32)
    qv_emb = query_embedding['v_emb'].to_torch().to(torch.float32)
    qs_emb = query_embedding['s_emb'].to_torch().to(torch.float32)
    transcriptions = rows['transcription_trocr']

    v_emb = rows['v3_emb'].to_torch().to(torch.float32) # 3 levels = v3_emb
    s_emb = rows['s_emb'].to_torch().to(torch.float32)

    result = {
        "q_emb": q_emb.expand(len(rows), -1), # [B, 38+384]
        "qv_emb": qv_emb.expand(len(rows), -1), # [B, 38]
        "qs_emb": qs_emb.expand(len(rows), -1), # [B, 384]
        "c_emb": rows['c_emb'].to_torch(),  # [B, 38+384]
        "v_emb": v_emb,  # [B, 38]
        "s_emb": s_emb,  # [B, 384]
        "v_sim": F.cosine_similarity(qv_emb, v_emb),
        "s_sim": F.cosine_similarity(qs_emb, s_emb),
        "rank_prior": rows['rank_prior'].to_torch(),
        "ocr_entropy": rows['ocr_entropy'].to_torch(),
        "levenshtein": torch.FloatTensor([levenshtein_similarity(query, t) for t in transcriptions]),
        "cer": torch.FloatTensor([compute_cer(query, t) for t in transcriptions]),
    }

    return {  k: ( v if v.dim() > 1 else v.unsqueeze(1) ).to(torch.float32)  for k,v in result.items()  }


def levenshtein_similarity(s1, s2):
    """Normalized Levenshtein similarity."""
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        result = 1.0
    else:
        result = 1.0 - (editdistance.eval(s1, s2) / max_len)
    return result

cer_metric = evaluate.load("cer")
def compute_cer(query, transcription):
    return cer_metric.compute(predictions=[transcription], references=[query])


