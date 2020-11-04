# telegram-analytics

A set of scripts for analyzing Telegram conversations written for personal use:

- `extract_messages.py`: produces a .csv file from a folder containing HTML 
   files for a single conversation; all the other scripts work on the files 
   produced by this script.
- `make_top_terms.py`:
   tokenizes messages and writes a JSON file with the top K terms 
   (stopword excluded) for each user in the chat.
- `make_word_clouds.py`: generates word clouds for:

   * the conversation (all users), 
   * each user,
   * each user relative to others.

   this particular script was tuned to my specific needs.
- `make_activity_plot.py`: generate a bar plot with the activity over time 
   measured by charachters/words/messages and aggregated by day/week/month/year.

Each script has a simple command line interface implemented using `argparse`. 
Check them out to see the usage and options.

## Dependencies
To install required packages run:
```shell
pip install -r requirements.txt
```
