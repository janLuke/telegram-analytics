"""
Generate a bar plot with the activity over time measured by
charachters/words/messages aggregated by day/week/month/year.
"""
from datetime import datetime
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import pandas as pd

VALID_METRICS = {'characters', 'words', 'messages'}
VALID_GROUPBY = {'day', 'week', 'month', 'year'}
DEFAULT_METRICS = ['messages']
DEFAULT_GROUPBY = 'month'


def date_as_datetime(value: Union[datetime, str]) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.strptime(value, '%Y-%m-%d')


def parse_datetime(string) -> datetime:
    return datetime.strptime(string, '%Y-%m-%d %H:%M:%S')


def get_activity(messages: pd.DataFrame,
                 from_date: Union[str, datetime, None] = None,
                 to_date: Union[str, datetime, None] = None,
                 metrics: str = DEFAULT_METRICS,
                 groupby: str = DEFAULT_GROUPBY) -> pd.DataFrame:
    if isinstance(metrics, str):
        metrics = [metrics]

    assert all(metric in VALID_METRICS for metric in metrics)
    assert groupby in VALID_GROUPBY

    if from_date is not None:
        from_date = date_as_datetime(from_date)
        messages = messages[messages.index >= from_date]

    if to_date is not None:
        to_date = date_as_datetime(to_date)
        messages = messages[messages.index <= to_date]

    grouper = {
        'day': lambda date: date.strftime('%Y-%m-%d'),
        'week': lambda date: date.strftime('%Y-%W'),
        'month': lambda date: date.strftime('%Y-%m'),
        'year': lambda date: date.strftime('%Y'),
    }[groupby]
    groups = messages.groupby(grouper)

    aggregate_by = {
        'characters': lambda msg_list: sum(len(msg) for msg in msg_list),
        'words': lambda msg_list: sum(len(msg) for msg in msg_list),
        'messages': 'count',
    }
    result = groups.aggregate(**{metric: ('text', aggregate_by[metric]) for metric in metrics})
    return result


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--messages_path')
    parser.add_argument('-o', '--out-path')
    parser.add_argument('--from-date', type=date_as_datetime, help='Start date')
    parser.add_argument('--to-date', type=date_as_datetime, help='End date')
    parser.add_argument('--metric', choices=VALID_METRICS, default=DEFAULT_METRICS[0])
    parser.add_argument('--groupby', choices=VALID_GROUPBY, default=DEFAULT_GROUPBY)
    parser.add_argument('--width', type=int, default=16)
    parser.add_argument('--height', type=int, default=6)

    args = parser.parse_args()
    args.messages_path = Path(args.messages_path)
    if not args.out_path:
        args.out_path = args.messages_path.parent / 'number_of_{}_by_{}.png'.format(args.metric,
                                                                                    args.groupby)

    messages = pd.read_csv(args.messages_path, index_col='datetime',
                           converters={'datetime': parse_datetime})
    activity = get_activity(messages, from_date=args.from_date, to_date=args.to_date,
                            metrics=args.metric, groupby=args.groupby)
    print(activity)

    plt.style.use('seaborn')
    fig = plt.figure(figsize=(args.width, args.height))
    plt.bar(activity.index, activity[args.metric])
    plt.xticks(rotation=45)
    plt.legend(activity.columns)
    xticks = range(0, len(activity), max(1, len(activity) // 30))
    plt.xticks(xticks)
    plt.xlabel(args.groupby.capitalize())
    plt.ylabel('Number of ' + args.metric)
    plt.title('Chat activity over time')
    plt.grid(axis='x')
    plt.tight_layout()
    plt.savefig(args.out_path)
