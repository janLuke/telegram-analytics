from concurrent.futures.process import ProcessPoolExecutor
from functools import partial
from pathlib import Path

from wordcloud import WordCloud

from make_top_terms import ChatTermFreq, make_top_terms_json


def save_wordcloud_image(path, freq, background_color="white", max_words=1000,
                         width=1024, height=768, skip_existing=False):
    if path.exists() and skip_existing:
        print('File already exists:', path.name)
        return

    print('Making word cloud', path.name, '...')
    max_words = max_words or len(freq)
    wc = WordCloud(
        background_color=background_color,
        max_words=max_words,
        width=width,
        height=height,
    )
    wc.generate_from_frequencies(freq)
    wc.to_file(path)
    print('Done word cloud', path.name)


def slugify(text, sep='_'):
    return sep.join(text.strip().split())


def make_word_clouds(out_dir, freqs: ChatTermFreq, max_words=1000,
                     width=1024, height=768, overwrite=False,
                     executor=None, num_workers=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    executor = executor or ProcessPoolExecutor(num_workers)
    save_wordcloud = partial(save_wordcloud_image,
        max_words=max_words, width=width, height=height, skip_existing=not overwrite)

    for key, counter in {'chat': freqs.of, **freqs.by_user}.items():
        out_path = out_dir / '{}_{}x{}.png'.format(slugify(key), width, height)
        executor.submit(save_wordcloud, out_path, counter)

    # For each user, compute the probability of using a term and keep the first 1000
    term_prob_by_user = {}
    for user, counter in freqs.by_user.items():
        total = sum(counter.values())
        term_prob_by_user[user] = {term: c / total for term, c in counter.most_common(1000)}

    # Compute how much each user is more likely to use a term wrt other users in the chat
    users = list(term_prob_by_user)
    n = len(users)
    for i, user in enumerate(users):
        others = users[:i] + users[i + 1:]
        user_prob = term_prob_by_user[user]
        others_prob = {term: sum(term_prob_by_user[name].get(term, 0) for name in others)
                       for term in user_prob}
        ratio = {term: n*p / (n*p + others_prob[term])
                 for term, p in user_prob.items()}
        ratio = {term: weight for term, weight in ratio.items() if weight > 1e-5}
        filename = 'Peculiar_of_{}_{}x{}.png'.format(slugify(user), width, height)
        executor.submit(save_wordcloud, out_dir / filename, ratio)

    executor.shutdown()


if __name__ == '__main__':
    import argparse
    from extract_messages import extract_messages_to_csv

    parser = argparse.ArgumentParser()
    parser.add_argument('chat_path', help='JSON file with top term frequencies')
    parser.add_argument('-W', '--width', default=1920, type=int, help='Word cloud width')
    parser.add_argument('-H', '--height', default=1080, type=int, help='Word cloud height')
    parser.add_argument('-m', '--max-words', default=1000, type=int,
        help='Max number of words in word cloud')
    parser.add_argument('--overwrite', default=False, action='store_true',
        help='Overwrite existing word clouds')
    parser.add_argument('-s', '--stemming', type=int, default=True, help='Use stemming')
    parser.add_argument('--langs', nargs='+', default=['italian', 'english'],
                        help='Languages used in chat')

    args = parser.parse_args()
    args.chat_path = Path(args.chat_path)

    messages_csv_path = args.chat_path / 'messages.csv'
    top_terms_path = args.chat_path / 'top_terms.json'

    if not messages_csv_path.exists():
        extract_messages_to_csv(args.chat_path, messages_csv_path)
    if not top_terms_path.exists():
        make_top_terms_json(messages_csv_path, top_terms_path,
                            stemming=args.stemming, languages=args.langs)

    with open(top_terms_path, encoding='utf-8', newline='') as f:
        freqs = ChatTermFreq.from_json(top_terms_path)

    make_word_clouds(
        out_dir=args.chat_path / 'word-clouds',
        freqs=freqs,
        width=args.width,
        height=args.height,
        max_words=args.max_words,
        overwrite=args.overwrite
    )
