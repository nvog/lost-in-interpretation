from sklearn.svm import SVC
import numpy as np
from feats import feature_fn
from reader import ModifiedConllCorpusReader
import os
import argparse
import random
from sklearn.metrics import classification_report, average_precision_score, precision_recall_fscore_support
from copy import deepcopy
from classifier_tagger import ClassifierTagger, ScaledSklearnClassifier


random.seed(666)
np.random.seed(666)
KW_TAG = 'I-K'
OUT_TAG = 'O'


def read_files(dir_path, file_paths):
    train_sents = []
    for file_path in file_paths:
        ccr = ModifiedConllCorpusReader(dir_path,
                          file_path,
                          ['words', 'pos', 'timing', 'chunk'])
        talk_position = 0
        for s, train_sent in enumerate(ccr.iob_sents()):
            sent = []
            for i, (word, pos, timing, chunk) in enumerate(train_sent):
                sent.append((word, pos, float(timing), talk_position, chunk))
                talk_position += 1
            train_sents.append(sent)
    return train_sents


def traverse_dir(directory, include_substrs=('',), exclude_substr=''):
    """
    Traverse a directory, only including the files with names that
    have the substring(s) include_substrs in them and files that do not
    contain exclude_substr in them.
    :param directory: str
    :param include_substrs: A tuple containing many possible substrings.
    :param exclude_substr: str
    :return: list of file paths
    """
    files = []
    for file in sorted(os.listdir(directory)):
        valid_file_name = True
        for fnc in include_substrs:
            if fnc not in file:
                valid_file_name = False
                break
        if valid_file_name and (exclude_substr == '' or exclude_substr not in file):
            files.append(os.path.join(directory, file))
    return files


def cv(train_data_path, test_data_path, output_file, rank, ablate=None, restricted_pos_set=None):
    train_data = np.array(traverse_dir(train_data_path, include_substrs=(rank,)))
    assert(train_data.shape[0] > 0)
    print('TRAIN:\n{}'.format(train_data))
    test_data = np.array(traverse_dir(test_data_path, include_substrs=(rank,)))
    assert(test_data.shape[0] > 0)
    print('TEST:\n{}'.format(test_data))

    all_untagged_sents = []
    all_true_tags = []
    all_pred_tags = []
    all_pred_scores = []
    outputf = open(output_file, 'w')

    for test_set_idx in range(len(train_data)):
        train_idxs = list(range(len(train_data)))
        train_idxs.remove(test_set_idx)
        dev_set_idx = (test_set_idx + 1) % len(train_idxs)
        train_idxs.remove(dev_set_idx)
        print('Starting CV {}'.format(test_set_idx + 1))
        train_path, dev_path, test_path = train_data[train_idxs], train_data[dev_set_idx], test_data[test_set_idx]
        print('train: {}\ndev: {}\ntest: {}'.format(train_path, dev_path, test_path))

        train_sents = read_files(train_data_path, [os.path.basename(p) for p in train_path])

        dev_sents = read_files(train_data_path, [os.path.basename(test_path)])
        best_tagger = None
        best_score = -1
        for c in [0.001, 0.01, 0.1, 1.0]:
            clf = ScaledSklearnClassifier(SVC(C=c, class_weight='balanced', kernel='linear', probability=True), sparse=False)
            tagger = ClassifierTagger(train=train_sents,
                                      classifier_builder=lambda train_feats: clf.train(train_feats),
                                      feature_detector=feature_fn, verbose=False, restricted_pos_set=restricted_pos_set,
                                      ablate=ablate)
            _, _, _, _, p, r, f, avg_precision = tagger.tag_sents(dev_sents)
            print('\tc={:0.4f}: P:{:.1f}, R:{:.1f}, F:{:.1f}, AP:{:.1f}'.format(c, p*100, r*100, f*100, avg_precision*100))
            if avg_precision > best_score:
                best_score = avg_precision
                best_tagger = deepcopy(tagger)

        test_sents = read_files(train_data_path, [os.path.basename(test_path)])
        untagged_sents, true_tags, pred_tags, pred_scores, p, r, f, avg_precision = best_tagger.tag_sents(test_sents)

        all_untagged_sents.extend(untagged_sents)
        all_true_tags.extend(true_tags)
        all_pred_tags.extend(pred_tags)
        all_pred_scores.extend(pred_scores)

        # filter by POS tag and output to file
        i = 0
        for sent in test_sents:
            for word_info in sent:
                if word_info[1] in restricted_pos_set:
                    print('\t'.join(list(map(str, list(word_info)[:1] + [pred_scores[i]] + list(word_info)[1:])) + [KW_TAG if pred_tags[i] == 1 else OUT_TAG]), file=outputf)
                    i += 1
                else:
                    print('\t'.join(list(map(str, list(word_info)[:1] + [0.0] + list(word_info)[1:])) + [OUT_TAG]), file=outputf)
            print('', file=outputf)

    outputf.close()
    print()

    print('TOTAL METRICS:')
    avg_precision = 100 * average_precision_score(all_true_tags, all_pred_scores)
    print('Avg. Precision: {:0.3f}'.format(avg_precision))
    print('Binary:', precision_recall_fscore_support(all_true_tags, all_pred_tags, average='binary'))

    print(classification_report(all_true_tags, all_pred_tags, digits=3))
    print()


def train_test(train_data_path, test_data_path, held_out_file_name, output_file, rank, ablate=None, restricted_pos_set=None):
    train_data = np.array(traverse_dir(train_data_path, include_substrs=(rank,), exclude_substr=held_out_file_name))
    assert(train_data.shape[0] > 0)
    print('TRAIN:\n{}'.format(train_data))
    test_data = np.array(traverse_dir(test_data_path, include_substrs=(rank, held_out_file_name)))
    assert(test_data.shape[0] > 0)
    print('TEST:\n{}'.format(test_data))

    all_untagged_sents = []
    all_true_tags = []
    all_pred_tags = []
    all_pred_scores = []
    outputf = open(output_file, 'w')

    train_path, dev_path, test_path = train_data[:-1], train_data[-1], test_data[0]

    print('train: {}\ndev: {}\ntest: {}'.format(train_path, dev_path, test_path))
    train_sents = read_files(train_data_path, [os.path.basename(p) for p in train_path])
    dev_sents = read_files(train_data_path, [os.path.basename(test_path)])
    best_tagger = None
    best_score = -1
    for c in [0.001, 0.01, 0.1, 1.0, 10.0]:
        clf = ScaledSklearnClassifier(SVC(C=c, class_weight='balanced', kernel='linear', probability=True), sparse=False)
        tagger = ClassifierTagger(train=train_sents,
                                  classifier_builder=lambda train_feats: clf.train(train_feats),
                                  feature_detector=feature_fn, verbose=True, restricted_pos_set=restricted_pos_set, ablate=ablate)
        _, _, _, _, p, r, f, avg_precision = tagger.tag_sents(dev_sents)
        print('\tc={:0.4f}: P:{:.1f}, R:{:.1f}, F:{:.1f}, AP:{:.1f}'.format(c, p*100, r*100, f*100, avg_precision*100))
        if avg_precision > best_score:
            best_score = avg_precision
            best_tagger = deepcopy(tagger)

    test_sents = read_files(train_data_path, [os.path.basename(test_path)])
    untagged_sents, true_tags, pred_tags, pred_scores, p, r, f, avg_precision = best_tagger.tag_sents(test_sents)

    all_untagged_sents.extend(untagged_sents)
    all_true_tags.extend(true_tags)
    all_pred_tags.extend(pred_tags)
    all_pred_scores.extend(pred_scores)

    # filter by POS tags and output to file
    i = 0
    for sent in test_sents:
        for word_info in sent:
            if word_info[1] in restricted_pos_set:
                print('\t'.join(list(map(str, list(word_info)[:1] + [pred_scores[i]] + list(word_info)[1:])) + [KW_TAG if pred_tags[i] == 1 else OUT_TAG]), file=outputf)
                i += 1
            else:
                print('\t'.join(list(map(str, list(word_info)[:1] + [0.0] + list(word_info)[1:])) + ['O']), file=outputf)
        print('', file=outputf)

    outputf.close()
    print()

    print('TOTAL METRICS:')
    avg_precision = 100 * average_precision_score(all_true_tags, all_pred_scores)
    print('Avg. Precision: {:0.3f}'.format(avg_precision))
    print('Binary:', precision_recall_fscore_support(all_true_tags, all_pred_tags, average='binary'))
    print(classification_report(all_true_tags, all_pred_tags, digits=3))

def setup(args):
    if args.cv:
        cv(args.train_dir, args.test_dir, args.output.format(args.rank), args.rank, ablate=args.ablate,
           restricted_pos_set={'CD', 'NN', 'NNS', 'NNP', 'NNPS'})
    elif args.train_test:
        train_test(args.train_dir, args.test_dir, args.train_test, args.output, args.rank,
                   ablate=args.ablate,
                   restricted_pos_set={'CD', 'NN', 'NNS', 'NNP', 'NNPS'})
    else:
        raise Exception('Must specify --train-test, --cv, or --predict.')


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filter', action='store_true')
    parser.add_argument('--load', default=None)
    parser.add_argument('--cv', action='store_true')
    parser.add_argument('--train-test', type=str,
                        help="Perform training on train data and testing on test data. Must specify a substring of the "
                             "file name to exclude from training set and include for test set a test file "
                             "(in combination with --rank) (e.g. 'AlGore').")
    parser.add_argument('--train-dir', default='data/subtitles/conll_timing')
    parser.add_argument('--test-dir', default='data/subtitles/conll_timing')
    parser.add_argument('--output', default='output.txt')
    parser.add_argument('--rank', default='Brank', choices=['Brank', 'Arank', 'Srank'])
    parser.add_argument('--ablate', default=None, type=str, choices=['word_frequency', 'characteristics/syntax', 'word_timing', 'elapsed_time'])
    args = parser.parse_args()
    print(args)
    return args


if __name__ == '__main__':
    args = parse_options()
    setup(args)
