import torch
import argparse

from tqdm import tqdm
from model.visil import ViSiL
from torch.utils.data import DataLoader
from datasets.generators import DatasetGenerator


@torch.no_grad()
def extract_features(model, frames, args):
    features = []
    for i in range(frames.shape[0] // args.batch_sz + 1):
        batch = frames[i*args.batch_sz: (i+1)*args.batch_sz]
        if batch.shape[0] > 0:
            features.append(model.extract_features(batch.to(args.gpu_id).float()))
    features = torch.cat(features, 0)
    while features.shape[0] < 4:
        features = torch.cat([features, features], 0)
    return features

@torch.no_grad()
def calculate_similarities_to_queries(model, queries, target, args):
    similarities = []
    for i, query in enumerate(queries):
        if query.device.type == 'cpu':
            query = query.to(args.gpu_id)
        sim = []
        for b in range(target.shape[0]//args.batch_sz_sim + 1):
            batch = target[b*args.batch_sz_sim: (b+1)*args.batch_sz_sim]
            if batch.shape[0] >= 4:
                sim.append(model.calculate_video_similarity(query, batch))
        sim = torch.mean(torch.cat(sim, 0))
        similarities.append(sim.cpu().numpy())
    return similarities 

def query_vs_target(model, dataset, args):
    # Create a video generator for the queries
    generator = DatasetGenerator(args.video_dir, dataset.get_queries(), args.pattern)
    loader = DataLoader(generator, num_workers=args.workers)

    # Extract features of the queries
    all_db, queries, queries_ids = set(), [], []
    print('> Extract features of the query videos')
    for video in tqdm(loader):
        frames = video[0][0]
        video_id = video[1][0]
        if frames.shape[0] > 0:
            features = extract_features(model, frames, args)
            if not args.load_queries: features = features.cpu()
            all_db.add(video_id)
            queries.append(features)
            queries_ids.append(video_id)

    # Create a video generator for the database video
    generator = DatasetGenerator(args.video_dir, dataset.get_database(), args.pattern)
    loader = DataLoader(generator, num_workers=args.workers)
    
    # Calculate similarities between the queries and the database videos
    similarities = dict({query: dict() for query in queries_ids})
    print('\n> Calculate query-target similarities')
    for video in tqdm(loader):
        frames = video[0][0]
        video_id = video[1][0]
        if frames.shape[0] > 0:
            features = extract_features(model, frames, args)
            sims = calculate_similarities_to_queries(model, queries, features, args)
            all_db.add(video_id)
            for i, s in enumerate(sims):
                similarities[queries_ids[i]][video_id] = float(s)
    
    print('\n> Evaluation on {}'.format(dataset.name))
    dataset.evaluate(similarities, all_db)


if __name__ == '__main__':
    formatter = lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, max_help_position=80)
    parser = argparse.ArgumentParser(description='This is the code for the evaluation of ViSiL network on five datasets.', formatter_class=formatter)
    parser.add_argument('--dataset', type=str, required=True, choices=["FIVR-200K", "FIVR-5K", "CC_WEB_VIDEO", "SVD", "EVVE"],
                        help='Name of evaluation dataset.')
    parser.add_argument('--video_dir', type=str, required=True,
                        help='Path to file that contains the database videos')
    parser.add_argument('--pattern', type=str, required=True,
                        help='Pattern that the videos are stored in the video directory, eg. \"{id}/video.*\" '
                             'where the \"{id}\" is replaced with the video Id. Also, it supports '
                             'Unix style pathname pattern expansion.')
    parser.add_argument('--batch_sz', type=int, default=128,
                        help='Number of frames contained in each batch during feature extraction.')
    parser.add_argument('--batch_sz_sim', type=int, default=2048,
                        help='Number of feature tensors in each batch during similarity calculation.')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='Id of the GPU used.')
    parser.add_argument('--load_queries', action='store_true',
                        help='Flag that indicates that the queries will be loaded to the GPU memory.')
    parser.add_argument('--similarity_function', type=str, default='chamfer', choices=["chamfer", "symmetric_chamfer"],
                        help='Function that will be used to calculate similarity '
                             'between query-target frames and videos.')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of workers used for video loading.')
    args = parser.parse_args()

    if 'CC_WEB' in args.dataset:
        from datasets import CC_WEB_VIDEO
        dataset = CC_WEB_VIDEO()
    elif 'FIVR' in args.dataset:
        from datasets import FIVR
        dataset = FIVR(version=args.dataset.split('-')[1].lower())
    elif 'EVVE' in args.dataset:
        from datasets import EVVE
        dataset = EVVE()
    elif 'SVD' in args.dataset:
        from datasets import SVD
        dataset = SVD()

    model = ViSiL(pretrained=True, symmetric='symmetric' in args.similarity_function).to(args.gpu_id)
    model.eval()

    query_vs_target(model, dataset, args)
