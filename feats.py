from nltk.corpus import cmudict
import inflect, re
import numpy as np

katakana_re = re.compile(r'^[ァ-ヴ・ヽヾ゛゜ー]+$')
number_re = re.compile(r'^(\d+|\d{1,3}(,\d{3})*)(\.\d+)?$')
cmu_dict = cmudict.dict()
inflect_engine = inflect.engine()
path_src_dict = 'data/dict/eiji-edict.clean.talkdicts.en'  # TODO: set to path of the EN side of your EIJIRO dictionary
path_tgt_dict = 'data/dict/eiji-edict.clean.talkdicts.ja'  # TODO: set to path of the JA side of your EIJIRO dictionary
path_word_freq_corpus = 'data/corpus/google1gram.en'       # TODO: set to path of your Google 1T Ngrams

# ****** FEATURE HYPERPARAMS ******
n_prev = 7
syllable_count_range = (1, 8)
n_syllable_count_bins = 7
word_freq_range = (1e-9, 1e-1)
n_word_freq_bins = 9
talk_position_range = (0, 3400)
n_talk_position_bins = 17
# *********************************


def load_dictionary(path_src_dict, path_tgt_dict):
    src_tgt_dict = dict()
    with open(path_src_dict, encoding='utf-8') as src, open(path_tgt_dict, encoding='utf-8') as tgt:
        for line in zip(src, tgt):
            src_entry = line[0].strip().lower()
            tgt_entry = line[1].strip().lower()
            if src_entry and tgt_entry:  # ensure neither entries are empty after stripping them of whitespace
                try:
                    src_tgt_dict[src_entry].append(tgt_entry)
                except KeyError:
                    src_tgt_dict[src_entry] = [tgt_entry]
    return src_tgt_dict


def load_word_freq_corpus(src_corpus_file):
    word_freq_counter = {}
    with open(src_corpus_file, encoding='utf-8') as f:
        for line in f:
            try:
                word, count = line.strip().split()
            except ValueError as e:
                continue
            word_freq_counter[word] = int(count)
    total = sum(word_freq_counter.values(), 0.0)
    for k in word_freq_counter.keys():
        word_freq_counter[k] /= total
    return word_freq_counter


src_tgt_dict = load_dictionary(path_src_dict, path_tgt_dict)
word_freq_counter = load_word_freq_corpus(path_word_freq_corpus)


def word_freq(word):
    try:
        word_freq = word_freq_counter[word]
    except KeyError:
        word_freq = word_freq_range[0]  # set to smallest possible word frequency
    if word_freq < word_freq_range[0]:  # place in lowest bin
        word_freq = word_freq_range[0]
    elif word_freq > word_freq_range[1]:
        word_freq = word_freq_range[1]
    bins, _ = np.histogram([word_freq], bins=np.logspace(np.log10(word_freq_range[0]),
                                                         np.log10(word_freq_range[1]), n_word_freq_bins))
    try:
        bin_index = np.where(bins)[0][0]
    except IndexError as e:
        print(str(word_freq), word)
        raise Exception(
            'Word: {}, Freq: {}. Try changing the --word-freq-range to fit the data.'.format(word, word_freq)
        ).with_traceback(e.__traceback__)
    return bin_index, word_freq


def talk_position(position):
    bins, _ = np.histogram([position], bins=n_talk_position_bins, range=talk_position_range)
    bin_index = np.where(bins)[0][0]
    return bin_index


def syllable_count(word):
    """
    from: https://stackoverflow.com/a/46759549
    and   https://stackoverflow.com/a/4103234
    """
    if len(word) == 0:
        return 0, 0

    word = word.lower()
    count = 0

    if word in cmu_dict:
        count = [len(list(y for y in x if y[-1].isdigit())) for x in cmu_dict[word]][0]
    elif number_re.match(word):  # convert to words
        for w in inflect_engine.number_to_words(word, andword='').replace(',', '').replace('-', ' ').split():
            count += [len(list(y for y in x if y[-1].isdigit())) for x in cmu_dict[w]][0]
    else:  # use simple heuristic
        count = 0
        vowels = "aeiouy"
        if word[0] in vowels:
            count += 1
        for index in range(1, len(word)):
            if word[index] in vowels and word[index - 1] not in vowels:
                count += 1
                if word.endswith("e"):
                    count -= 1
        if count == 0:
            count += 1
    if count >= syllable_count_range[1]:
        count = syllable_count_range[1] - 1
    bins, _ = np.histogram([count], bins=n_syllable_count_bins, range=syllable_count_range)
    bin_index = np.where(bins)[0][0]
    return bin_index, count


def is_number(word):
    return int(bool(number_re.match(word)))


def has_katakana_entry(en_word):
    try:
        entry = src_tgt_dict[en_word.lower()]
        for s in entry:
            if katakana_re.match(s):
                return 1
    except KeyError:
        pass
    return 0


def calc_n_words_past_secs(timings, index, n_secs):
    count = 0
    for time in reversed(timings[:index]):
        if time >= timings[index] - n_secs:
            count += 1
    return count


def feature_fn(tokens, pos_tags, timings, talk_positions, index, history, ablate=None):
    """

    :param tokens: list of unlabeled tokens in the sentence with other info in a tuple
    :param pos_tags: list of POS tags for each word
    :param timings: list of start times for each word
    :param talk_positions: list of talk position indices for each word
    :param index: index of the token for which feature detection should be performed
    :param history: history is list of the (predicted_tag, pos_tag) for all tokens before index
    :param ablate: set of features to ablate
    :return: feature set
    """
    feature_set = {}

    # pad
    ptokens = ('',) * n_prev + tokens + ('',)
    ppos_tags = ('<s>',) * n_prev + pos_tags + ('</s>',)
    phistory = ['<s>',] * n_prev + history
    # history of predicted tags
    prev_pred_tags = {'prev_pred_tags-' + str(i): phistory[-i] for i in range(1, n_prev+1)}
    feature_set.update(prev_pred_tags)

    curr_word = tokens[index]
    prev_word = ptokens[n_prev + index - 1]
    curr_time = timings[index]

    # word frequency
    if ablate != 'word_frequency':
        feature_set.update({
            'word_freq': str(word_freq(curr_word)[0]),
            'loan_word': str(has_katakana_entry(curr_word)),   # TODO: ONLY LOOK IN FIRST THREE DEFINITIONS?
        })

    # word characteristics + syntax
    if ablate != 'characteristics/syntax':
        feature_set.update({
            'curr_pos_tag': pos_tags[index],
            'word_len': len(curr_word),
            'syll_count': syllable_count(curr_word)[1],
            'number': str(is_number(curr_word)),
            'prev_number': str(is_number(prev_word)),
        })
        prev_pos_tags = {'prev_pos_tags-' + str(i): ppos_tags[n_prev + index - i] for i in range(1, n_prev+1)}
        prev_word_lens = {'prev_word_lens-' + str(i): len(ptokens[n_prev + index - i]) for i in range(1, n_prev+1)}
        prev_word_sylls = {'prev_word_sylls-' + str(i): syllable_count(ptokens[n_prev + index - i])[1] for i in range(1, n_prev+1)}
        feature_set.update(prev_pos_tags)
        feature_set.update(prev_word_lens)
        feature_set.update(prev_word_sylls)

    # word timing
    if ablate != 'word_timing':
        feature_set.update({
            'n_words_past_2_secs': calc_n_words_past_secs(timings, index, 2),
            'n_words_past_4_secs': calc_n_words_past_secs(timings, index, 4),
            'n_words_past_7_secs': calc_n_words_past_secs(timings, index, 7),
            'n_words_past_10_secs': calc_n_words_past_secs(timings, index, 10),
            'prev_time_delta': curr_time - timings[index - 1] if index > 0 else 0.,
            'prev_prev_time_delta': curr_time - timings[index - 2] if index > 1 else 0.
        })

    # elapsed time
    if ablate != 'elapsed_time':
        feature_set.update({
            'sent_index': index,
            'talk_position': talk_positions[index],
            'n_mins_elapsed_talk': curr_time / 60.,
        })

    return feature_set
