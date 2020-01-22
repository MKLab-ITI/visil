import argparse
import tensorflow as tf

from tqdm import tqdm
from model.visil import ViSiL
from datasets import DatasetGenerator


def query_vs_database(model, dataset, args):
    # Create a video generator for the queries
    enqueuer = tf.keras.utils.OrderedEnqueuer(
        DatasetGenerator(args.video_dir, dataset.get_queries(), args.pattern, all_frames='i3d' in args.network),
        use_multiprocessing=True, shuffle=False)
    enqueuer.start(workers=args.threads, max_queue_size=args.threads * 2)

    # Extract features of the queries
    all_db, queries, queries_ids = set(), [], []
    pbar = tqdm(range(len(enqueuer.sequence)))
    for _ in pbar:
        frames, query_id = next(enqueuer.get())
        if frames.shape[0] > 0:
            queries.append(model.extract_features(frames, batch_sz=25 if 'i3d' in args.network else args.batch_sz))
            queries_ids.append(query_id)
            all_db.add(query_id)
            pbar.set_postfix(query_id=query_id)
    enqueuer.stop()
    model.set_queries(queries)

    # Create a video generator for the database video
    enqueuer = tf.keras.utils.OrderedEnqueuer(
        DatasetGenerator(args.video_dir, dataset.get_database(), args.pattern, all_frames='i3d' in args.network),
        use_multiprocessing=True, shuffle=False)
    enqueuer.start(workers=args.threads, max_queue_size=args.threads * 2)
    generator = enqueuer.get()

    # Calculate similarities between the queries and the database videos
    similarities = dict({query: dict() for query in queries_ids})
    pbar = tqdm(range(len(enqueuer.sequence)))
    for _ in pbar:
        frames, video_id = next(generator)
        if frames.shape[0] > 1:
            features = model.extract_features(frames, batch_sz=25 if 'i3d' in args.network else args.batch_sz)
            sims = model.calculate_similarities_to_queries(features)
            all_db.add(video_id)
            for i, s in enumerate(sims):
                similarities[queries_ids[i]][video_id] = float(s)
            pbar.set_postfix(video_id=video_id)
    enqueuer.stop()

    dataset.evaluate(similarities, all_db)


def all_vs_all(model, dataset, args):
    # Create a video generator for the dataset video
    enqueuer = tf.keras.utils.OrderedEnqueuer(
        DatasetGenerator(args.video_dir, dataset.get_queries(), args.pattern, all_frames='i3d' in args.network),
        use_multiprocessing=True, shuffle=False)
    enqueuer.start(workers=args.threads, max_queue_size=args.threads * 2)

    # Calculate similarities between all videos in the dataset
    all_db, similarities, features = set(), dict(), dict()
    pbar = tqdm(range(len(enqueuer.sequence)))
    for _ in pbar:
        frames, q = next(enqueuer.get())
        if frames.shape[0] > 0:
            all_db.add(q)
            similarities[q] = dict()
            feat = model.extract_features(frames, batch_sz=25 if 'i3d' in args.network else args.batch_sz)
            for k, v in features.items():
                if 'symmetric' in args.similarity_function:
                    similarities[q][k] = similarities[k][q] = model.calculate_video_similarity(v, feat)
                else:
                    similarities[k][q] = model.calculate_video_similarity(v, feat)
                    similarities[q][k] = model.calculate_video_similarity(feat, v)
            features[q] = feat
            pbar.set_postfix(video_id=q, frames=frames.shape, features=feat.shape)
    enqueuer.stop()

    dataset.evaluate(similarities, all_db=all_db)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, required=True,
                        help='Name of evaluation dataset. Options: CC_WEB_ VIDEO, '
                             '\"FIVR-200K\", \"FIVR-5K\", \"EVVE\", \"ActivityNet\"')
    parser.add_argument('-v', '--video_dir', type=str, required=True,
                        help='Path to file that contains the database videos')
    parser.add_argument('-p', '--pattern', type=str, required=True,
                        help='Pattern that the videos are stored in the video directory, eg. \"{id}/video.*\" '
                             'where the \"{id}\" is replaced with the video Id. Also, it supports '
                             'Unix style pathname pattern expansion.')
    parser.add_argument('-n', '--network', type=str, default='resnet',
                        help='Backbone network used for feature extraction. '
                             'Options: \"resnet\" or \"i3d\". Default: \"resnet\"')
    parser.add_argument('-m', '--model_dir', type=str, default='ckpt/resnet',
                        help='Path to the directory of the pretrained model. Default: \"ckpt/resnet\"')
    parser.add_argument('-s', '--similarity_function', type=str, default='chamfer',
                        help='Function that will be used to calculate similarity'
                             'between query-candidate frames and videos.'
                             'Options: \"chamfer\" or \"symmetric_chamfer\". Default: \"chamfer\"')
    parser.add_argument('-b', '--batch_sz', type=int, default=128,
                        help='Number of frames contained in each batch during feature extraction. Default: 128')
    parser.add_argument('-g', '--gpu_id', type=int, default=0,
                        help='Id of the GPU used. Default: 0')
    parser.add_argument('-l', '--load_queries', action='store_true',
                        help='Flag that indicates that the queries will be loaded to the GPU memory.')
    parser.add_argument('-t', '--threads', type=int, default=8,
                        help='Number of threads used for video loading. Default: 8')
    args = parser.parse_args()

    if 'CC_WEB' in args.dataset:
        from datasets import CC_WEB_VIDEO
        dataset = CC_WEB_VIDEO()
        eval_function = query_vs_database
    elif 'FIVR' in args.dataset:
        from datasets import FIVR
        dataset = FIVR(version=args.dataset.split('-')[1].lower())
        eval_function = query_vs_database
    elif 'EVVE' in args.dataset:
        from datasets import EVVE
        dataset = EVVE()
        eval_function = query_vs_database
    elif 'ActivityNet' in args.dataset:
        from datasets import ActivityNet
        dataset = ActivityNet()
        eval_function = all_vs_all
    else:
        raise Exception('[ERROR] Not supported evaluation dataset. '
                        'Supported options: \"CC_WEB_ VIDEO\", \"FIVR-200K\", \"FIVR-5K\", \"EVVE\", \"ActivityNet\"')

    model = ViSiL(args.model_dir, net=args.network,
                  load_queries=args.load_queries, gpu_id=args.gpu_id,
                  similarity_function=args.similarity_function,
                  queries_number=len(dataset.get_queries()) if args.load_queries else None)

    eval_function(model, dataset, args)
