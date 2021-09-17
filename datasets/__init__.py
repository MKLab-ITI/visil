import os
import cv2
import glob
import numpy as np
import pickle as pk
import tensorflow as tf


def resize_frame(frame, desired_size):
    min_size = np.min(frame.shape[:2])
    ratio = desired_size / min_size
    frame = cv2.resize(frame, dsize=(0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)
    return frame


def center_crop(frame, desired_size):
    old_size = frame.shape[:2]
    top = int(np.maximum(0, (old_size[0] - desired_size)/2))
    left = int(np.maximum(0, (old_size[1] - desired_size)/2))
    return frame[top: top+desired_size, left: left+desired_size, :]


def load_video(video, all_frames=False):
    cv2.setNumThreads(3)
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps > 144 or fps is None:
        fps = 25
    frames = []
    count = 0
    while cap.isOpened():
        ret = cap.grab()
        if int(count % round(fps)) == 0 or all_frames:
            ret, frame = cap.retrieve()
            if isinstance(frame, np.ndarray):
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(center_crop(resize_frame(frame, 256), 256))
            else:
                break
        count += 1
    cap.release()
    return np.array(frames)


class VideoGenerator(tf.keras.utils.Sequence):
    def __init__(self, video_file, all_frames=False):
        super(VideoGenerator, self).__init__()
        self.videos = np.loadtxt(video_file, dtype=str)
        self.videos = np.expand_dims(self.videos, axis=0) if self.videos.ndim == 1 else self.videos
        self.all_frames = all_frames

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        return load_video(self.videos[index][1], all_frames=self.all_frames), self.videos[index][0]


class DatasetGenerator(tf.keras.utils.Sequence):
    def __init__(self, rootDir, videos, pattern, all_frames=False):
        super(DatasetGenerator, self).__init__()
        self.rootDir = rootDir
        self.videos = videos
        self.pattern = pattern
        self.all_frames = all_frames

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        video = glob.glob(os.path.join(self.rootDir, self.pattern.replace('{id}', self.videos[index])))
        if not len(video):
            print('[WARNING] Video not found: ', self.videos[index])
            return np.array([]), None
        else:
            return load_video(video[0], all_frames=self.all_frames), self.videos[index]


class CC_WEB_VIDEO(object):

    def __init__(self):
        with open('datasets/cc_web_video.pickle', 'rb') as f:
            dataset = pk.load(f)
        self.database = dataset['index']
        self.queries = dataset['queries']
        self.ground_truth = dataset['ground_truth']
        self.excluded = dataset['excluded']

    def get_queries(self):
        return self.queries

    def get_database(self):
        return list(map(str, self.database.keys()))

    def calculate_mAP(self, similarities, all_videos=False, clean=False, positive_labels='ESLMV'):
        mAP = 0.0
        for query_set, labels in enumerate(self.ground_truth):
            query_id = self.queries[query_set]
            i, ri, s = 0.0, 0.0, 0.0
            if query_id in similarities:
                res = similarities[query_id]
                for video_id in sorted(res.keys(), key=lambda x: res[x], reverse=True):
                    video = self.database[video_id]
                    if (all_videos or video in labels) and (not clean or video not in self.excluded[query_set]):
                        ri += 1
                        if video in labels and labels[video] in positive_labels:
                            i += 1.0
                            s += i / ri
                positives = np.sum([1.0 for k, v in labels.items() if
                                    v in positive_labels and (not clean or k not in self.excluded[query_set])])
                mAP += s / positives
        return mAP / len(set(self.queries).intersection(similarities.keys()))

    def evaluate(self, similarities, all_db=None):
        if all_db is None:
            all_db = self.database

        print('=' * 5, 'CC_WEB_VIDEO Dataset', '=' * 5)
        not_found = len(set(self.queries) - similarities.keys())
        if not_found > 0:
            print('[WARNING] {} queries are missing from the results and will be ignored'.format(not_found))
        print('Queries: {} videos'.format(len(similarities)))
        print('Database: {} videos'.format(len(all_db)))

        print('-' * 25)
        print('All dataset')
        print('CC_WEB mAP: {:.4f}\nCC_WEB* mAP: {:.4f}\n'.format(
            self.calculate_mAP(similarities, all_videos=False, clean=False),
            self.calculate_mAP(similarities, all_videos=True, clean=False)))

        print('Clean dataset')
        print('CC_WEB mAP: {:.4f}\nCC_WEB* mAP: {:.4f}'.format(
            self.calculate_mAP(similarities, all_videos=False, clean=True),
            self.calculate_mAP(similarities, all_videos=True, clean=True)))


class FIVR(object):

    def __init__(self, version='200k'):
        self.version = version
        with open('datasets/fivr.pickle', 'rb') as f:
            dataset = pk.load(f)
        self.annotation = dataset['annotation']
        self.queries = dataset[self.version]['queries']
        self.database = dataset[self.version]['database']

    def get_queries(self):
        return self.queries

    def get_database(self):
        return list(self.database)

    def calculate_mAP(self, query, res, all_db, relevant_labels):
        gt_sets = self.annotation[query]
        query_gt = set(sum([gt_sets[label] for label in relevant_labels if label in gt_sets], []))
        query_gt = query_gt.intersection(all_db)

        i, ri, s = 0.0, 0, 0.0
        for video in sorted(res.keys(), key=lambda x: res[x], reverse=True):
            if video != query and video in all_db:
                    ri += 1
                    if video in query_gt:
                        i += 1.0
                        s += i / ri
        return s / len(query_gt)

    def evaluate(self, similarities, all_db=None):
        if all_db is None:
            all_db = self.database

        DSVR, CSVR, ISVR = [], [], []
        for query, res in similarities.items():
            if query in self.queries:
                DSVR.append(self.calculate_mAP(query, res, all_db, relevant_labels=['ND', 'DS']))
                CSVR.append(self.calculate_mAP(query, res, all_db, relevant_labels=['ND', 'DS', 'CS']))
                ISVR.append(self.calculate_mAP(query, res, all_db, relevant_labels=['ND', 'DS', 'CS', 'IS']))

        print('=' * 5, 'FIVR-{} Dataset'.format(self.version.upper()), '=' * 5)
        not_found = len(set(self.queries) - similarities.keys())
        if not_found > 0:
            print('[WARNING] {} queries are missing from the results and will be ignored'.format(not_found))

        print('Queries: {} videos'.format(len(similarities)))
        print('Database: {} videos'.format(len(all_db)))

        print('-' * 16)
        print('DSVR mAP: {:.4f}'.format(np.mean(DSVR)))
        print('CSVR mAP: {:.4f}'.format(np.mean(CSVR)))
        print('ISVR mAP: {:.4f}'.format(np.mean(ISVR)))


class EVVE(object):

    def __init__(self):
        with open('datasets/evve.pickle', 'rb') as f:
            dataset = pk.load(f)
        self.events = dataset['annotation']
        self.queries = dataset['queries']
        self.database = dataset['database']
        self.query_to_event = {qname: evname
                               for evname, (queries, _, _) in self.events.items()
                               for qname in queries}

    def get_queries(self):
        return list(self.queries)

    def get_database(self):
        return list(self.database)

    def score_ap_from_ranks_1(self, ranks, nres):
        """ Compute the average precision of one search.
        ranks = ordered list of ranks of true positives (best rank = 0)
        nres  = total number of positives in dataset
        """
        if nres == 0 or ranks == []:
            return 0.0

        ap = 0.0

        # accumulate trapezoids in PR-plot. All have an x-size of:
        recall_step = 1.0 / nres

        for ntp, rank in enumerate(ranks):
            # ntp = nb of true positives so far
            # rank = nb of retrieved items so far

            # y-size on left side of trapezoid:
            if rank == 0:
                precision_0 = 1.0
            else:
                precision_0 = ntp / float(rank)
            # y-size on right side of trapezoid:
            precision_1 = (ntp + 1) / float(rank + 1)
            ap += (precision_1 + precision_0) * recall_step / 2.0
        return ap

    def evaluate(self, similarities, all_db=None):
        results = {e: [] for e in self.events}
        if all_db is None:
            all_db = set(self.database).union(set(self.queries))

        not_found = 0
        for query in self.queries:
            if query not in similarities:
                not_found += 1
            else:
                res = similarities[query]
                evname = self.query_to_event[query]
                _, pos, null = self.events[evname]
                if all_db:
                    pos = pos.intersection(all_db)
                pos_ranks = []

                ri, n_ext = 0.0, 0.0
                for ri, dbname in enumerate(sorted(res.keys(), key=lambda x: res[x], reverse=True)):
                    if dbname in pos:
                        pos_ranks.append(ri - n_ext)
                    if dbname not in all_db:
                        n_ext += 1

                ap = self.score_ap_from_ranks_1(pos_ranks, len(pos))
                results[evname].append(ap)

        print('=' * 18, 'EVVE Dataset', '=' * 18)

        if not_found > 0:
            print('[WARNING] {} queries are missing from the results and will be ignored'.format(not_found))
        print('Queries: {} videos'.format(len(similarities)))
        print('Database: {} videos\n'.format(len(all_db - set(self.queries))))
        print('-' * 50)
        ap = []
        for evname in sorted(self.events):
            queries, _, _ = self.events[evname]
            nq = len(queries.intersection(all_db))
            ap.extend(results[evname])
            print('{0: <36} '.format(evname), 'mAP = {:.4f}'.format(np.sum(results[evname]) / nq))

        print('=' * 50)
        print('overall mAP = {:.4f}'.format(np.mean(ap)))


class ActivityNet(object):

    def __init__(self):
        with open('datasets/activity_net.pickle', 'rb') as f:
            self.dataset = pk.load(f)

    def get_queries(self):
        return list(map(str, self.dataset.keys()))

    def get_database(self):
        return list(map(str, self.dataset.keys()))

    def calculate_AP(self, res, pos):
        i, ri, s = 0.0, 0.0, 0.0
        for ri, video in enumerate(sorted(res.keys(), key=lambda x: res[x], reverse=True)):
            if video in pos:
                i += 1.0
                s += i / (ri + 1.)
        return s / len(pos)

    def evaluate(self, similarities, all_db=None):
        mAP, not_found = [], 0
        if all_db is None:
            all_db = set(self.get_database())

        for query in self.dataset.keys():
            if query not in similarities:
                not_found += 1
            else:
                pos = self.dataset[query].intersection(all_db)
                mAP += [self.calculate_AP(similarities[query], pos)]

        print('=' * 5, 'ActivityNet Dataset', '=' * 5)
        if not_found > 0:
            print('[WARNING] {} queries are missing from the results and will be ignored'.format(not_found))
        print('Database: {} videos'.format(len(all_db)))

        print('-' * 16)
        print('mAP: {:.4f}'.format(np.mean(mAP)))
