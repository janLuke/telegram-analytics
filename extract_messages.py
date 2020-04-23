"""
Given the path of an exported chat (HTML), extracts all messages and writes
them to a .csv file.
"""
import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from itertools import chain, islice
from pathlib import Path
from typing import Iterator, List, Optional

import bs4
from tqdm import tqdm

logging.basicConfig(format='[%(levelname)s] %(message)s', level='INFO')


@dataclass
class Message:
    datetime: datetime
    username: str
    text: str


def extract_messages_from_file(path: Path) -> List[Message]:
    """ Extracts messages from a messages{index}.html file. """
    messages = []
    html = path.read_text(encoding='utf-8')
    parser = bs4.BeautifulSoup(html, 'html.parser')
    username = None
    for div in parser.select('.message.default'):
        if 'joined' not in div['class']:
            username_elem = div.select_one('.from_name')
            if username_elem:
                username = username_elem.text.strip()
        if username is None:
            raise Exception('no username')

        # Skip the message if it doesn't contain any text or it's a forwarded message
        text_elem = div.select_one('.body:not(forwarded) > .text')
        if text_elem is None:
            continue
        text = text_elem.text.strip()

        datetime_string = div.select_one('.date')['title']
        parsed_datetime = datetime.strptime(datetime_string, '%d.%m.%Y %H:%M:%S')

        messages.append(
            Message(username=username, datetime=parsed_datetime, text=text))
    return messages


def get_message_page_paths(chat_dir: Path) -> List[Path]:
    """ Return a sorted list of message pages """
    n = len('messages')
    def get_index(path: Path):  # messages{index}.html
        return int(path.stem[n:] or '0')
    return sorted(chat_dir.glob('messages*.html'), key=get_index)


class HtmlChatParser:
    def __init__(self, chat_dir: Path, num_workers: Optional[int] = None):
        self.chat_dir = chat_dir
        self._executor = ProcessPoolExecutor(num_workers)
        self.page_paths = get_message_page_paths(chat_dir)
        self.num_pages = len(self.page_paths)

    def parse_pages(self, limit: Optional[int] = None) -> Iterator[List[Message]]:
        file_paths = list(islice(self.page_paths, 0, limit))
        return self._executor.map(extract_messages_from_file, file_paths)

    def shutdown(self):
        self._executor.shutdown()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()


def write_messages_to_csv(out_path: Path, messages: Iterator[Message]):
    import csv

    with open(out_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['datetime', 'username', 'text'])
        for msg in messages:
            writer.writerow([msg.datetime, msg.username, msg.text])
    logging.info('CSV file written to %s', out_path)


def extract_messages_to_csv(chat_dir: Path,
                            out_path: Optional[Path] = None,
                            limit: Optional[int] = None,
                            num_workers: Optional[int] = None):
    if not out_path:
        out_path = Path(chat_dir, 'messages.csv')

    with HtmlChatParser(chat_dir, num_workers=num_workers) as chat_parser:
        page_iterator = chat_parser.parse_pages(limit=limit)

        message_list = chain.from_iterable(
            tqdm(
                page_iterator,
                total=limit or chat_parser.num_pages,
                desc='Parsing chat pages',
                unit='pages'
            ))
        write_messages_to_csv(out_path, message_list)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'chat_dir',
        help='Directory containing the exported chat')
    parser.add_argument(
        '-o', '--out-path',
        help='Output file path; by default, the file is saved in chat-dir as messages.csv')
    parser.add_argument(
        '-l', '--limit', type=int,
        help='Max number of chat files (1000 messages each)')
    parser.add_argument(
        '-n', '--num-workers', type=int,
        help='Number of concurrent processes to use for message extraction')

    args = parser.parse_args()
    args.chat_dir = Path(args.chat_dir)
    extract_messages_to_csv(
        chat_dir=args.chat_dir,
        out_path=args.out_path,
        limit=args.limit,
        num_workers=1
    )
