import argparse
import numpy as np
import pandas as pd

gamma = 2
sizes = [5, 11, 20, 35, 80]
seqsignet_sizes = [(3, 5, 3), (3, 5, 6), (3, 5, 11), (3, 5, 26)]

def readout_results_from_csv(csv_filename: str, model_name: str, digits: int):
    try:
        results_df = pd.read_csv(csv_filename)
        print(f"{'#'*10} {model_name} {'#'*10}")
        # print overall F1, precision and recall scores
        print(f"F1: {round(results_df['f1'].mean(), digits)}")
        print(f"Precision: {round(results_df['precision'].mean(), digits)}")
        print(f"Recall: {round(results_df['recall'].mean(), digits)}")
        # print individual class F1, precision and recall scores averaged
        f1_scores_stacked = np.stack(results_df['f1_scores'].apply(lambda x: list(np.fromstring(x[1:-1], sep=' '))))
        print(f"F1 scores: {[round(x, digits) for x in f1_scores_stacked.mean(axis=0)]}")
        precision_scores_stacked = np.stack(results_df['precision_scores'].apply(lambda x: list(np.fromstring(x[1:-1], sep=' '))))
        print(f"Precision scores: {[round(x, digits) for x in precision_scores_stacked.mean(axis=0)]}")
        recall_scores_stacked = np.stack(results_df['recall_scores'].apply(lambda x: list(np.fromstring(x[1:-1], sep=' '))))
        print(f"Recall scores: {[round(x, digits) for x in recall_scores_stacked.mean(axis=0)]}")
        print("\n")
    except:
        print(f"Error reading {csv_filename}")

def main():
     # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        "-r",
        help="Path to the folder containing the results data.",
        type=str,
    )
    parser.add_argument(
        "--digits",
        "-d",
        help="digits to round to",
        type=int,
        default=3,
    )
    args = parser.parse_args()

    # these variables are manually set for now:
    gamma = 2
    sizes = [5, 11, 20, 35]
    seqsignet_sizes = [(3, 5, 3), (3, 5, 6), (3, 5, 11)]
    
    # readout FFN
    file = f"{args.results_dir}/ffn_current_focal_{gamma}_kfold_best_model.csv"
    readout_results_from_csv(file, model_name="FFN with current", digits=args.digits)

    # readout FFN with history concatenation
    file = f"{args.results_dir}/ffn_mean_history_focal_{gamma}_kfold_best_model.csv"
    readout_results_from_csv(file, model_name="FFN with mean history concatenated with current", digits=args.digits)

    # readout BERT (focal loss)
    file = f"{args.results_dir}/bert_classifier_focal.csv"
    readout_results_from_csv(file, model_name="BERT (focal)", digits=args.digits)

    # readout BERT (ce)
    file = f"{args.results_dir}/bert_classifier_ce.csv"
    readout_results_from_csv(file, model_name="BERT (cross-entropy)", digits=args.digits)

    # readout LSTM
    for size in sizes:
        file = f"{args.results_dir}/lstm_history_{size}_focal_{gamma}_kfold_best_model.csv"
        readout_results_from_csv(file, model_name=f"BiLSTM (size={size})", digits=args.digits)
        
    # readout SWNU-Network
    for size in sizes:
        file = f"{args.results_dir}/swnu_network_umap_focal_{gamma}_{size}_kfold_best_model.csv"
        readout_results_from_csv(file, model_name=f"SWNU-Network (size={size})", digits=args.digits)

    # readout SWMHAU-Network
    for size in sizes:
        file = f"{args.results_dir}/swmhau_network_umap_focal_{gamma}_{size}_kfold_best_model.csv"
        readout_results_from_csv(file, model_name=f"SWMHAU-Network (size={size})", digits=args.digits)

    # readout SeqSigNet
    for shift, window_size, n in seqsignet_sizes:
        file = f"{args.results_dir}/seqsignet_umap_focal_{gamma}_{shift}_{window_size}_{n}_kfold_best_model.csv"
        k = shift * n + (window_size - shift)
        readout_results_from_csv(file, model_name=f"SeqSigNet (size={k})", digits=args.digits)

    # readout SeqSigNetAttentionBiLSTM
    for shift, window_size, n in seqsignet_sizes:
        file = f"{args.results_dir}/seqsignet_attention_bilstm_umap_focal_{gamma}_{shift}_{window_size}_{n}_kfold_best_model.csv"
        k = shift * n + (window_size - shift)
        readout_results_from_csv(file, model_name=f"SeqSigNetAttentionBiLSTM (size={k})", digits=args.digits)
    
    # readout SeqSigNetAttentionEncoder
    for shift, window_size, n in seqsignet_sizes:
        file = f"{args.results_dir}/seqsignet_attention_encoder_umap_focal_{gamma}_{shift}_{window_size}_{n}_kfold_best_model.csv"
        k = shift * n + (window_size - shift)
        readout_results_from_csv(file, model_name=f"SeqSigNetAttentionEncoder (size={k})", digits=args.digits)

if __name__ == "__main__":
    main()