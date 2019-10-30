import json
import argparse
import tensorflow as tf

from tqdm import tqdm
from model import ViSiL
from dataset import VideoGenerator

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-q', '--query_file', type=str, required=True,
                        help='Path to file that contains the query videos')
    parser.add_argument('-d', '--database_file', type=str, required=True,
                        help='Path to file that contains the database videos')
    parser.add_argument('-o', '--output_file', type=str, default='results.json',
                        help='Name of the output file. Default: \"results.json\"')
    parser.add_argument('-m', '--model_dir', type=str, default='model/',
                        help='Path to the directory of the pretrained model. Default: \"model/\"')
    parser.add_argument('-g', '--gpu_id', type=int, default=0,
                        help='Id of the GPU used. Default: 0')
    parser.add_argument('-l', '--load_queries', action='store_true',
                        help='Flag that indicates that the queries will be loaded to the GPU memory.')

    args = parser.parse_args()

    # Create a video generator for the queries
    enqueuer = tf.keras.utils.OrderedEnqueuer(VideoGenerator(args.query_file),
                                              use_multiprocessing=True, shuffle=False)
    enqueuer.start(workers=10, max_queue_size=20)
    generator = enqueuer.get()

    # Initialize ViSiL model
    model = ViSiL(args.model_dir, load_queries=args.load_queries, gpu_id=args.gpu_id,
                  queries_number=len(enqueuer.sequence) if args.load_queries else None)

    # Extract features of the queries
    queries, queries_ids = [], []
    pbar = tqdm(range(len(enqueuer.sequence)))
    for _ in pbar:
        frames, video_id = next(generator)
        features = model.extract_features(frames, 100)
        queries.append(features)
        queries_ids.append(video_id)
        pbar.set_postfix(query_id=video_id)
    enqueuer.stop()
    model.set_queries(queries)

    # Create a video generator for the database video
    enqueuer = tf.keras.utils.OrderedEnqueuer(VideoGenerator(args.database_file),
                                              use_multiprocessing=True, shuffle=False)
    enqueuer.start(workers=10, max_queue_size=20)
    generator = enqueuer.get()

    # Calculate similarities between the queries and the database videos
    similarities = dict({query: dict() for query in queries_ids})
    pbar = tqdm(range(len(enqueuer.sequence)))
    for _ in pbar:
        frames, video_id = next(generator)
        if frames.shape[0] > 1:
            sims = model.calculate_similarities(frames, 100)
            for i, s in enumerate(sims):
                similarities[queries_ids[i]][video_id] = float(s)
            pbar.set_postfix(video_id=video_id)
    enqueuer.stop()

    # Save similarities to a json file
    with open(args.output_file, 'w') as f:
        json.dump(similarities, f, indent=1)
