import json
import torch
import argparse

from tqdm import tqdm
from model.visil import ViSiL
from torch.utils.data import DataLoader
from datasets.generators import VideoGenerator
from evaluation import extract_features, calculate_similarities_to_queries


if __name__ == '__main__':
    formatter = lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, max_help_position=80)
    parser = argparse.ArgumentParser(description='This is the code for video similarity calculation based on ViSiL network.', formatter_class=formatter)
    parser.add_argument('--query_file', type=str, required=True,
                        help='Path to file that contains the query videos')
    parser.add_argument('--database_file', type=str, required=True,
                        help='Path to file that contains the database videos')
    parser.add_argument('--output_file', type=str, default='results.json',
                        help='Name of the output file.')
    parser.add_argument('--batch_sz', type=int, default=128,
                        help='Number of frames contained in each batch during feature extraction. Default: 128')
    parser.add_argument('--batch_sz_sim', type=int, default=2048,
                        help='Number of feature tensors in each batch during similarity calculation.')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='Id of the GPU used.')
    parser.add_argument('--load_queries', action='store_true',
                        help='Flag that indicates that the queries will be loaded to the GPU memory.')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of workers used for video loading.')
    args = parser.parse_args()

    # Create a video generator for the queries
    generator = VideoGenerator(args.query_file)
    loader = DataLoader(generator, num_workers=args.workers)

    # Initialize ViSiL model
    model = ViSiL(pretrained=True).to(args.gpu_id)
    model.eval()

    # Extract features of the queries
    queries, queries_ids = [], []
    pbar = tqdm(loader)
    for video in pbar:
        frames = video[0][0]
        video_id = video[1][0]
        features = extract_features(model, frames, args)
        if not args.load_queries: features = features.cpu()
        queries.append(features)
        queries_ids.append(video_id)
        pbar.set_postfix(query_id=video_id)

    # Create a video generator for the database video
    generator = VideoGenerator(args.database_file)
    loader = DataLoader(generator, num_workers=args.workers)

    # Calculate similarities between the queries and the database videos
    similarities = dict({query: dict() for query in queries_ids})
    pbar = tqdm(loader)
    for video in pbar:
        frames = video[0][0]
        video_id = video[1][0]
        if frames.shape[0] > 1:
            features = extract_features(model, frames, args)
            sims = calculate_similarities_to_queries(model, queries, features, args)
            for i, s in enumerate(sims):
                similarities[queries_ids[i]][video_id] = float(s)
            pbar.set_postfix(video_id=video_id)

    # Save similarities to a json file
    with open(args.output_file, 'w') as f:
        json.dump(similarities, f, indent=1)
