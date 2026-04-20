import numpy as np
import polars as pl
import torch
import torchvision.ops as tvops



def compute_relevance_in_page(bboxes, gt_bboxes, iou_threshold):
    if gt_bboxes is None:
        return np.zeros(len(bboxes), dtype=np.float32)
    if bboxes is None:
        return np.zeros(0, dtype=np.float32)

    num_bboxes = len(bboxes)
    relevant_instances = np.zeros(num_bboxes, dtype=np.float32)

    num_gt_bboxes = len(gt_bboxes)
    seen_gt_bboxes = np.zeros(num_gt_bboxes, dtype=np.float32)

    i = 0
    while i < num_bboxes and seen_gt_bboxes.sum() < num_gt_bboxes:
        iou, reverse_index = tvops.box_iou(
            torch.Tensor(bboxes[np.newaxis, i]),
            torch.Tensor(gt_bboxes)
        ).squeeze(0).sort(descending=True)

        j = 0
        while j < num_gt_bboxes and iou[j] > iou_threshold:
            gt_j = reverse_index[j]
            if not seen_gt_bboxes[gt_j]:
                seen_gt_bboxes[gt_j] = 1
                relevant_instances[i] = 1
                break
            j += 1
        i += 1

    assert relevant_instances.sum() <= num_gt_bboxes

    return relevant_instances

def compute_relevance(suggestion_df, ground_truth_df, subset, iou_threshold=0.25):
    suggestion_df = suggestion_df.with_row_index()
    grouped_suggestion_dict = {(q,p):df for (q,p), df in suggestion_df.group_by(["query", "page"])}
    ground_truth_dict = {(q,p):df for (q,p), df in ground_truth_df.filter(subset=subset).group_by(["query", "page"])}

    ids = []
    relevant = []

    for (q,p), query_df in grouped_suggestion_dict.items():
        ids.extend(query_df['index'].to_list())

        if p is None: # the query had an empty ranked list
            relevant.extend([None]*len(query_df))
            continue

        if (q,p) not in ground_truth_dict: # no suggestions occurred on the current sunset of pages
            relevant.extend([False]*len(query_df))
            continue

        bboxes = query_df['bbox'].to_numpy()
        gt_bboxes = ground_truth_dict[(q,p)]['gt_bbox'].to_numpy()

        relevant_in_page = compute_relevance_in_page(
            np.vstack(bboxes),
            np.vstack(gt_bboxes),
            iou_threshold
        ).astype(bool).tolist()

        relevant.extend(relevant_in_page)

    return suggestion_df.join(
        pl.DataFrame({
            "index": pl.Series(ids).cast(pl.UInt32),
            "relevant": pl.Series(relevant)
        }),
        on="index",
        how="full",
        coalesce=True
    ).drop('index')




def compute_average_precision(relevant_instances, actual_number_of_relevant_instances):
    assert relevant_instances.ndim == 1 and relevant_instances.sum() <= actual_number_of_relevant_instances

    relevant_instances_at_k_retrievals = np.cumsum(relevant_instances, dtype=float)
    number_of_retrievals = 1 + np.arange(len(relevant_instances))
    precision_at_k_retrievals = relevant_instances_at_k_retrievals / number_of_retrievals

    if actual_number_of_relevant_instances>0:
        average_precision = (precision_at_k_retrievals * relevant_instances).sum() / actual_number_of_relevant_instances
    else:
        average_precision = 0

    return average_precision


def rerank_by(suggestions, key):
    if f'ranking_{key}' in suggestions.columns:
        suggestions = suggestions.drop('ranking')
    return suggestions.with_columns(
        (pl.col(key).rank("ordinal", descending=True)
                             .over("query") - 1)
                             .alias(f'ranking_{key}')
    )


def compute_map_of_reranked_by(key, suggestions, ground_truth, iou_threshold, *, keep_topk=30):
    return compute_map(
        rerank_by(suggestions, key),
        f"ranking_{key}",
        ground_truth,
        iou_threshold,
        keep_topk=keep_topk
    )


def compute_map(suggestion_df, ranking_key, ground_truth_df, iou_threshold, *, keep_topk=30):
    average_precision_per_query = []

    # queries for which the system has no retrievals
    average_precision_per_query.extend(
        [0.0]*len(suggestion_df.filter(pl.col('bbox').is_null()))
    )

    suggestion_df = (
        suggestion_df.filter(pl.col(ranking_key) < keep_topk)
                     .sort(ranking_key)
                     .group_by(["query", "page"])
                     .all()
    )

    ground_truth_df = ground_truth_df.filter(
        pl.col("query").is_in(suggestion_df["query"]),
        pl.col("page").is_in(suggestion_df["page"]),
    ).group_by(["query", "page"]).all()

    joined_df = suggestion_df.join(
        ground_truth_df,
        on=["query","page"],
        how="full",
        coalesce=True
    )

    for _, query_df in joined_df.group_by("query"):
        if query_df["bbox"].is_null().all():
            print(query_df['query'].unique().item())
            continue

        relevant_instances = np.zeros(
            query_df.select(pl.col('bbox').list.len().sum()).item(),
            dtype=np.float32
        )

        for gt_bboxes, bboxes, ranking in query_df["gt_bbox", "bbox", ranking_key].iter_rows():
            if bboxes is None or gt_bboxes is None:
                continue

            relevant_instances[ranking] = compute_relevance_in_page(
                np.vstack(bboxes),
                np.vstack(gt_bboxes),
                iou_threshold
            )

        number_of_gt_bboxes = (
            query_df.select( pl.col("gt_bbox").list.len() )
                    .sum()
                    .item()
        )

        number_of_gt_bboxes = min(keep_topk, number_of_gt_bboxes)

        if not (relevant_instances.sum() <= number_of_gt_bboxes):
            print(query_df)
            print(relevant_instances)
            print(number_of_gt_bboxes)

        assert relevant_instances.sum() <= number_of_gt_bboxes

        try:
            average_precision = compute_average_precision(relevant_instances, number_of_gt_bboxes)
        except AssertionError:
            print(query_df)
            print(relevant_instances)
            print(number_of_gt_bboxes)
            raise
        average_precision_per_query.append(average_precision)

    return np.mean(average_precision_per_query)



def compute_map_of_ideal_reranking(suggestion_df, subset, ground_truth_df):
    """Computes the mAP of the ideal reranking - i.e. all relevant results are return first"""
    return compute_map_of_reranked_by('relevant',suggestion_df.with_columns(pl.col('relevant').cast(pl.Float32)),ground_truth_df,0.25)

    ground_truth_by_query = { q: df_q for (q,), df_q in ground_truth_df.filter(subset=subset).group_by('query') }
    ap = []
    for (q,), qdf in suggestion_df.filter(relevant=True).group_by('query'):
        r = len(qdf) # all the found relevant instances
        n = len(ground_truth_by_query[q]) # all the existing relevant instances
        assert r<=n
        ap.append(r/n) # average precision simplifies to this expression
    return np.mean(ap)
