import argparse
import logging
from typing import Tuple

import yaml

from get_hayate import GetHayate
from line_notify import LineNotify


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('log/log.log')
file_handler.setLevel(logging.INFO)
fh_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s - %(name)s - %(funcName)s - %(message)s')
file_handler.setFormatter(fh_formatter)
logger.addHandler(file_handler)

def parse():
    parser = argparse.ArgumentParser('Get images from Twitter and post LINE')
    parser.add_argument('-m', '--model_path', default='streamlit/models/hayate_finetune.pt', help='Path for pytorch model')
    parser.add_argument('-s', '--secret_path', default='secret.yaml', help='Path for secret')
    parser.add_argument('-c', '--config_path', default='config.yaml', help='Path for config')
    args = parser.parse_args()
    return args

def load_yamls(secret_path='secret.yaml', config_path='config.yaml') -> Tuple[dict, dict]:
    with open(secret_path, 'r', encoding="utf-8") as f:
        secret = yaml.safe_load(f)
        
    with open(config_path, 'r', encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return secret, config

def make_query(config: dict) -> str:
    query = config['words']
    query += ' filter:images'
    query += f' min_faves:{config["min_faves"]}'
    if config['retweets_exclude']:
        query += ' exclude:retweets'
    return query
        
def set_apis(twitter_token, line_token, model_path: str, save_path: str):
    hayate = GetHayate(twitter_token, model_path, save_path)
    line_api = LineNotify(line_token['token'], line_token['notify_api'])
    return hayate, line_api

def main(twitter_api, line_api, twitter_config: dict) -> None:
    query = make_query(twitter_config)
    items = twitter_config['items']
    logger.info({'action': 'main', 'config':twitter_config})
    
    line_api.send(msg=f'\n検索ワード：{query}\n検索数：{items}')
    tweets = hayate.get_tweets(query=query, result_type='recent', items=items)
    msg = '\n'.join(tweets)
    line_api.send(msg=msg, image=open(save_path, 'rb'))
    logger.info({'action': 'main', 'status': 'success'})
    
if __name__ == '__main__':
    args = parse()

    model_path = args.model_path
    secret_path = args.secret_path
    config_path = args.config_path
    secret, config = load_yamls(secret_path, config_path)

    twitter_config = config['twitter']
    save_path = twitter_config['save_path']
    
    twitter_token = secret['twitter']
    line_token = secret['line']
    hayate, line_api = set_apis(twitter_token, line_token, model_path, save_path)
    
    main(hayate, line_api, twitter_config)