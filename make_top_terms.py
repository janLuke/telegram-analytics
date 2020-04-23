"""
Given a .csv file containing messages (datetime, user, text), tokenize messages
by author and writes a JSON file with the top K terms (stopword excluded) for
each author and for the full chat.
"""
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import ClassVar, Dict, Iterable, Optional, List

import nltk
import wordcloud
from nltk import SnowballStemmer
from tqdm.auto import tqdm

from extract_messages import Message

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

EXTRA_EN_STOPWORDS = ["n't", 'yes', "'ll", "'ve", 'must', 'much', 'would', "'re", '...']
EXTRA_IT_STOPWORDS = ['cosa']
STOPWORDS = wordcloud.STOPWORDS | set(
    nltk.corpus.stopwords.words('english') + EXTRA_EN_STOPWORDS
    + nltk.corpus.stopwords.words('italian') + EXTRA_IT_STOPWORDS
)

URL_PATTERN = re.compile(
    r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')


def iterable_len(iterable: Iterable, default=None):
    try:
        return len(iterable)
    except TypeError:
        return default


def clean_text(text: str) -> str:
    return URL_PATTERN.sub('', text.replace("'", ' ')).lower()


@dataclass
class ChatTermFreq:
    JSON_FIELD__CHAT_FREQ: ClassVar[str] = 'chat_term_frequencies'
    JSON_FIELD__FREQ_BY_USER: ClassVar[str] = 'term_frequencies_by_user'

    of: Counter  # term -> frequency in the chat (all users)
    by_user: Dict[str, Counter]  # user -> {term -> frequency relative to the user}

    def __getitem__(self, term) -> int:
        return self.of[term]

    def to_json(self, path, top=2000):
        file_content = {
            self.JSON_FIELD__CHAT_FREQ: dict(self.of.most_common(top)),
            self.JSON_FIELD__FREQ_BY_USER: {
                user: dict(counter.most_common(top))
                for user, counter in self.by_user.items()
            }
        }
        with open(path, 'w', encoding='utf-8', newline='') as f:
            json.dump(file_content, f, ensure_ascii=False, indent=2)

    @staticmethod
    def from_json(path) -> 'ChatTermFreq':
        with open(path, encoding='utf-8', newline='') as f:
            content = json.load(f)
        return ChatTermFreq(
            Counter(content[ChatTermFreq.JSON_FIELD__CHAT_FREQ]),
            {user: Counter(freqs) for user, freqs in
             content[ChatTermFreq.JSON_FIELD__FREQ_BY_USER].items()}
        )


class StemmerWrapper:
    def __init__(self, stem_func):
        self._stem = stem_func
        self.stemmed2term_freq: Dict[str, Counter] = defaultdict(Counter)

    def stem(self, term):
        stemmed = self._stem(term)
        self.stemmed2term_freq[stemmed][term] += 1
        return stemmed

    def most_common_form_of(self, stemmed):
        return self.stemmed2term_freq[stemmed].most_common()[0][0]


def _get_term_frequencies(messages: Iterable[Message]) -> ChatTermFreq:
    user_freq = defaultdict(Counter)
    chat_freq = Counter()

    for msg in tqdm(messages, unit='message', total=iterable_len(messages),
                    desc='Counting term frequencies'):
        cleaned_msg = clean_text(msg.text)
        tokens = [token for token in nltk.word_tokenize(cleaned_msg)
                 if len(token) > 2 and token not in STOPWORDS]
        if not tokens:
            continue
        msg_freq = Counter(tokens)
        user_freq[msg.username].update(msg_freq)
        chat_freq.update(msg_freq)

    return ChatTermFreq(chat_freq, user_freq)


def replace_terms(term2count: Counter, replacer: Dict[str, str]) -> Counter:
    return Counter({replacer[stemmed]: count for stemmed, count in term2count.items()})


def _get_term_frequencies_with_stemming(messages: Iterable[Message],
                                        languages: List[str]) -> ChatTermFreq:
    assert len(languages) > 0
    user_freq = defaultdict(Counter)
    chat_freq = Counter()

    stemmers = [SnowballStemmer(lang).stem for lang in languages]
    def stemmatize(term):
        for f in stemmers:
            term = f(term)
        return term

    stemmer = StemmerWrapper(stemmatize)
    all_stems = set()

    for msg in tqdm(messages, unit='message', total=iterable_len(messages),
                    desc='Counting term frequencies'):
        cleaned_msg = clean_text(msg.text)
        stems = [stemmer.stem(token) for token in nltk.word_tokenize(cleaned_msg)
                 if len(token) > 2 and token not in STOPWORDS]
        if not stems:
            continue
        msg_freq = Counter(stems)
        user_freq[msg.username].update(msg_freq)
        chat_freq.update(msg_freq)
        all_stems.update(stems)

    to_most_common_form = {stem: stemmer.most_common_form_of(stem) for stem in all_stems}
    for user in user_freq:
        user_freq[user] = replace_terms(user_freq[user], to_most_common_form)
    chat_freq = replace_terms(chat_freq, to_most_common_form)
    return ChatTermFreq(chat_freq, user_freq)


def get_term_frequencies(messages: Iterable[Message],
                         stemming: bool = False,
                         languages=['english']) -> ChatTermFreq:
    if stemming:
        return _get_term_frequencies_with_stemming(messages, languages)
    else:
        return _get_term_frequencies(messages)


def make_top_terms_json(messages_csv: Path, out_path: Path,
                        top: int = 2000,
                        limit: Optional[int] = None,
                        skip_bots: bool = True,
                        stemming: bool = False,
                        languages=['english']) -> Dict[str, Counter]:
    with open(messages_csv, encoding='utf-8', newline='') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        messages = (Message(date, username, text) for date, username, text in reader)
        if skip_bots:
            messages = (msg for msg in messages if ' via @' not in msg.username)
        if limit is not None:
            messages = islice(messages, 0, limit)
        term_freqs = get_term_frequencies(list(messages), stemming=stemming, languages=languages)
        term_freqs.to_json(out_path, top=top)
        return term_freqs


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('messages_path', help='CSV file with messages')
    parser.add_argument('-o', '--out-path', default=None,
                        help='Output file path; by default, the file is saved in the same directory of messages-path '
                             'as top_terms.json')
    parser.add_argument('-t', '--top', default=2000,
                        help='Number of most frequent words to keep (for each user and for the total)')
    parser.add_argument('-l', '--limit', type=int, default=None,
                        help='Max number of messages to read')
    parser.add_argument('-s', '--stemming', type=int, default=True, help='Use stemming')
    parser.add_argument('--langs', nargs='+', default=['english', 'italian'],
                        help='Languages used in chat')

    args = parser.parse_args()
    args.messages_path = Path(args.messages_path)
    args.out_path = Path(args.out_path or (args.messages_path.parent / 'top_terms.json'))

    make_top_terms_json(
        args.messages_path, args.out_path,
        top=args.top,
        limit=args.limit,
        skip_bots=True,
        stemming=args.stemming,
        languages=args.langs
    )
