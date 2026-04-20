import argparse
from pathlib import Path


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train or evaluate an attention-based kws re-ranking model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--mode",
        choices=["train", "eval"],
        default='eval',
        help="Choose oen between training and evaluation mode"
    )

    parser.add_argument(
        "--iam-path",
        type=Path,
        required=True,
        help="Path to IAM (Parquet file containing queries, bounding boxes, embeddings, etc.)"
    )

    parser.add_argument(
    "--kws-output-train",
        type=Path,
        help=(
            "Path to the KWS ranked lists over the queries of the train pages"
            "(Parquet file containing queries, bounding boxes, embeddings, etc.)"
        )
    )

    parser.add_argument(
    "--kws-output-validation",
        type=Path,
        help=(
            "Path to the KWS ranked lists over the queries of the validation pages"
            "(Parquet file containing queries, bounding boxes, embeddings, etc.)"
        )
    )

    parser.add_argument(
    "--kws-output-test",
        type=Path,
        required=True,
        help=(
            "Path to the KWS ranked lists over the queries of the test pages"
            "(Parquet file containing queries, bounding boxes, embeddings, etc.)"
        )
    )

    parser.add_argument(
        "--save-path",
        type=Path,
        help="Path to save trained model"
    )

    parser.add_argument(
        "--load-path",
        type=Path,
        help="Path to load model checkpoint"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate"
    )

    parser.add_argument(
        "--negative-ratio",
        type=int,
        default=10,
        help="Negative sampling ratio"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Examples per batch"
    )

    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.25,
        help="IoU threshold"
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=30,
        help="Keep the top-k suggestions from each ranked list"
    )

    parser.add_argument(
        "--experts",
        nargs="+",
        choices=["scalar", "ocr", "vec"],
        default=["scalar", "ocr", "vec"],
        help="Enabled experts"
    )

    return parser


parser = build_parser()
args = parser.parse_args()


from evaluation import *
from map_metric import compute_map_of_reranked_by, compute_map_of_ideal_reranking
from dataset import load_iam_ground_truth, load_ranked_lists, PairwiseDataset, TripletDataset
from reranker import QueryConditionedTokenMixer
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup



print(f'> Running script configuration: {args}')


print('> Loading ground truth')
ground_truth = load_iam_ground_truth(args.iam_path)
query_embedding_dict = produce_query_embeddings_dict(ground_truth)


print('> Loading datasets')
test_suggestions = load_ranked_lists(args.kws_output_test, ground_truth, 'test', iou_threshold=args.iou_threshold)

if args.mode == "train":
    train_suggestions = load_ranked_lists(args.kws_output_train, ground_truth, 'train', iou_threshold=args.iou_threshold)
    val_suggestions = load_ranked_lists(args.kws_output_validation, ground_truth, 'validation', iou_threshold=args.iou_threshold)
print('> Finished loading datasets')


mAP_at_k = f"mAP@{int(args.iou_threshold*100)}"

print('> Computing recall at 25 -> ./R@25_v_sim.png')
plot_recalls(25, test_suggestions, 'v_sim')
print(f'> {mAP_at_k} of ideal reranking:', compute_map_of_ideal_reranking(test_suggestions, 'test', ground_truth))
print(f'> {mAP_at_k} of visual reranking:', compute_map_of_reranked_by('v_sim', test_suggestions, ground_truth, args.iou_threshold))



#TODO model configurations,loading, etc
query_specs=(("qv_emb", 228), ("qs_emb", 384)) #(q_emb, 228+384)
if 'scalar' in args.experts:
    scalar_keys=("v_sim", "s_sim", "rank_prior")
else:
    scalar_keys=()
if 'ocr' in args.experts:
    ocr_keys=("ocr_entropy", "levenshtein", "cer")
else:
    ocr_keys=()
if 'vec' in args.experts:
    vec_specs=(("s_emb", 384), ('v_emb_tok', 38))#("v_emb", 228), ('c_emb', 228+384))
else:
    vec_specs=()



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"> Using: {device}")


model = QueryConditionedTokenMixer(d=512, d_vec=128, query_specs=query_specs, scalar_keys=scalar_keys, ocr_keys=ocr_keys, vec_specs=vec_specs)
model = model.to(device)
if args.load_path is not None:
    model = torch.load(args.load_path, map_location=device)



def report_map_of_learned_scores(msg, suggestions_df):
    suggestions_df = compute_learned_scores(model, suggestions_df, query_embedding_dict)
    map_ = compute_map_of_reranked_by(
        'learned_score',
        suggestions_df['query','bbox','page','learned_score'],
        ground_truth,
        args.iou_threshold,
        keep_topk = args.top_k
    )
    print(msg, map_)
    return map_, suggestions_df


if args.mode == "train":
    train_set = PairwiseDataset(train_suggestions, query_embedding_dict, neg_ratio=args.negative_ratio)
    # train_set = TripletDataset(train_suggestions, query_embedding_dict, neg_ratio=args.negative_ratio)

    dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    output = model({k:v.to(device) for k,v in train_set[0][-1].items()})
    print("> Sanity check of batch input:", output['final_score'].shape)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.01)
    num_training_steps = args.epochs * len(dataloader)
    num_warmup_steps = int(0.05 * num_training_steps)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    ########################################################################################################################
    ########################################################################################################################
    ########################################################################################################################
    #TODO which loss function?

    def pairwise_softplus_loss(s_pos, s_neg):
        # log(1 + exp(-(Δ))) = softplus(-Δ)
        return F.softplus(-(s_pos - s_neg)).mean()

    # criterion = torch.nn.BCEWithLogitsLoss()
    # def pairwise_bce_loss(pos, neg):
    #     logits = pos - neg
    #     targets = torch.ones_like(logits)
    #     return criterion(logits, targets)

    margin_loss = nn.MarginRankingLoss(0.2)

    ########################################################################################################################
    ########################################################################################################################
    ########################################################################################################################

    import sys
    import time
    epoch=0
    def save_training_state(*_args):
        timestamp = int(time.time())
        if args.save_path is None:
            out = Path('./output/tmp.pt')
        else:
            out = args.save_path
        try:
            print(f'Saving model state at {out.parent/out.name.replace(".pt",f"_{timestamp}.pt")}')
            with open(f'{out.parent/out.name.replace(".pt", f"_{timestamp}.txt")}', 'w') as f:
                print('args: ', { **vars(args), 'epoch': epoch, 'loss': 'margin_loss' }, file=f)
                print('mAPs: ', mAPs, file=f)
            torch.save(model, f'{out.parent/out.name.replace(".pt",f"_{timestamp}.pt")}')
        except NameError:
            pass
        sys.__excepthook__(*_args)

    sys.excepthook = save_training_state


    entropy_max = 0.05
    entropy_warmup_epochs = 10

    loss_coeff = {
        'final_score': 1,
        'scalar_score': 0.2,
        'ocr_score': 0.2,
        'vec_score': 0.2,
    }
    losses_keys = loss_coeff.keys()

    mAPs = {'epoch': [], 'validation':[]}

    mAPs['epoch'].append(0)
    map_ = 0
    map_,val_suggestions = report_map_of_learned_scores(f"> Initial {mAP_at_k} on validation set: ", val_suggestions)
    mAPs['validation'].append(map_)



    print("> Started training")
    for epoch in tqdm(range(1, args.epochs+1), desc="Epochs"):
        total_losses = { k: 0 for k in losses_keys }
        total_loss = 0

        entropy_weight = entropy_max * min(epoch / entropy_warmup_epochs, 1.0)
        entropy = 0

        diffs = []
        contibutions = []

        if (epoch>=6 and epoch % 3 == 0):
            print("> Mining hard negatives...")
            train_set.mine_hard_negatives(lambda df: compute_learned_scores(model, df, query_embedding_dict))
            dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

        model.train()
        for feats_pos, feats_neg in tqdm(dataloader, leave=False, desc="Batches", position=0):

            feats_pos = {
                k: v.reshape(-1,v.shape[2]).to(device)
                    for k,v in feats_pos.items()
            }
            out_pos = model(feats_pos)

            feats_neg = {
                k: v.reshape(-1,v.shape[2]).to(device) # 32*neg_ratio; the input sequence is flattern along the first dims
                    for k,v in feats_neg.items()
            }
            out_neg = model(feats_neg)

            target = torch.ones_like(out_pos['final_score']).to(device)  # indicates pos should be ranked higher in MarginRankingLoss
            batch_losses = {k: loss_coeff[k]*margin_loss(out_pos[k], out_neg[k], target) for k in losses_keys}

            loss = torch.zeros([]).to(device)
            for k in losses_keys:
                loss += batch_losses[k]
                total_losses[k] += batch_losses[k].item()

            if epoch < 10:
                entropy = -(out_pos['weights'] * torch.log(out_pos['weights'] + 1e-8)).sum(dim=-1).mean()
                entropy = - entropy_weight * entropy
                loss += entropy_weight * entropy # negative entropy
            else:
                entropy = 0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            diffs.append((out_pos['final_score']-out_neg['final_score']).mean())
            contibutions.append((out_pos['weights'].mean(dim=0)+out_neg['weights'].mean(dim=0)))

        print(f"> Finished training {epoch=} with losses:")
        print(pl.DataFrame({
                'epoch': epoch,
                'loss': total_loss,
                **dict([ (k,round(v,4))  for k,v in total_losses.items() ]),
                "entropy": entropy
        }))

        print(f"> Mean (positive - negative) scores =", torch.stack(diffs).mean().item())

        print(f'> Mean gate behaviour {[round(i,4) for i in torch.stack(contibutions).mean(dim=0).tolist()]}')

        if epoch%3==0:
            model.eval()
            mAPs['epoch'].append(epoch)
            map_,val_suggestions = report_map_of_learned_scores(f"> {mAP_at_k} on validation set: ", val_suggestions)
            mAPs['validation'].append(map_)
            save_loc = str(args.save_path).replace('.pt',f'_{epoch}.pt')
            print(f"Saving model state at {save_loc}")
            torch.save(model, save_loc)

    print(pl.DataFrame(mAPs))
    pl.DataFrame(mAPs).write_csv(args.save_path.parent/args.save_path.name.replace('.pt','_mAPs.csv'))

    print("The training has finished")

report_map_of_learned_scores(f"> {mAP_at_k} on test set: ", test_suggestions)
