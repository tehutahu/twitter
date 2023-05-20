import io
import logging
import os
import traceback
from typing import List
from typing import Tuple

from PIL import Image
from PIL.JpegImagePlugin import JpegImageFile
import numpy as np
import requests
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torchvision.utils import make_grid
import tweepy


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('log/log.log')
file_handler.setLevel(logging.INFO)
fh_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s - %(name)s - %(funcName)s - %(message)s')
file_handler.setFormatter(fh_formatter)
logger.addHandler(file_handler)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.ERROR)
sh_formatter = logging.Formatter('%(asctime)s - %(message)s', '%Y-%m-%d %H:%M:%S')
stream_handler.setFormatter(sh_formatter)
logger.addHandler(stream_handler)

class GetHayate(object):
    
    CLASS_NAMES = ['official', 'user']
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __init__(self, tokens: dict, model_path: str, save_path='grid.jpg') -> None:
        self.api = self.api_auth(tokens)
        self.model = self.build_model(model_path)
        self.save_path = save_path
        self.tweets = None
    
    def api_auth(self, tokens: dict) -> tweepy.API:
        consumer_key = tokens['consumer_key']
        consumer_secret = tokens['consumer_secret']
        access_token = tokens['access_token']
        access_token_secret = tokens['access_token_secret']
        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        api = tweepy.API(auth)
        logger.info({'action': 'api_auth', 'status': 'success'})
        return api
    
    def build_model(self, model_path: str) -> torch.nn.Module:
        model = torchvision.models.resnet50()
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        m_state_dict = torch.load(model_path, map_location=self.DEVICE)
        model.load_state_dict(m_state_dict)
        model.to(self.DEVICE)
        model.eval()
        logger.info({'action': 'build_model', 'status': 'success'})
        return model

    def inference(self, image: JpegImageFile) -> Tuple[str, float]:
        try:
            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            image_t = preprocess(image)
            batch_t = torch.unsqueeze(image_t, 0)
            batch_t = batch_t.to(self.DEVICE)
            
            out = self.model(batch_t)
            _, index = torch.max(out, 1)
            per = nn.functional.softmax(out, dim=1)[0] * 100
        except Exception as e:
            logger.error({'action': 'inference', 'error': e})
            raise e
        else:
            logger.info({'action': 'inference', 'result': self.CLASS_NAMES[index[0]]})
            return self.CLASS_NAMES[index[0]], per[index[0]].item()
    
    def is_user_original(self, image: JpegImageFile) -> bool:
        try:
            result = self.inference(image)[0] == self.CLASS_NAMES[1]
        except Exception as e:
            logger.error({'action': 'inference', 'error': e})
            result = False
        return result
        
    def save_image_grid(self, images: List[JpegImageFile], size=(1280, 1280)) -> List[str]:
        def make_tensor(image: JpegImageFile, size=(240, 240)) -> torch.Tensor:
            tensor = torch.from_numpy(np.array(image).transpose(2, 0, 1))
            tensor = transforms.Resize(size)(tensor)
            return tensor
        n_imgs = 25
        paths = []
        tensors = list(map(make_tensor, images))
        for i in range(0, len(tensors), n_imgs):
            path = os.path.join(self.save_path, f'{i}.png')
            grid_arr = make_grid(tensors[i:i+n_imgs], nrow=5).numpy().transpose(1, 2, 0)
            Image.fromarray(grid_arr).save(path)
            paths.append(path)
        logger.info({'action': 'save_image_grid', 'status': 'success'})
        return paths
    
    def get_tweets(self, query="#久川颯 OR 久川颯 exclude:retweets filter:images min_faves:50",
                    result_type='mixed', items=100) -> Tuple[List[str], List[str]]:
        tweets = []
        images = []
        for status in tweepy.Cursor(self.api.search_tweets, query, result_type=result_type).items(items):
            if hasattr(status, 'extended_entities') and 'media' in status.extended_entities:
                for media in status.extended_entities['media']:
                    tweet_url = media['url']
                    # tweet_url = status.entities['media'][0]['url']
                    image_url = media['media_url']
                    # image_url = status.entities['media'][0]['media_url']
                    response = requests.get(image_url, stream=True)
                    if response.status_code == 200:
                        try:
                            image = Image.open(io.BytesIO(response.raw.data)).convert('RGB')
                        except Exception:
                            logger.error({'action': 'get_tweets', 'err': traceback.format_exc()}) 
                            continue
                        if self.is_user_original(image):
                            tweets.append(tweet_url)
                            images.append(image)
                            logger.info({'action': 'get_tweets', 'accept_url': tweet_url})
                        else:
                            logger.info({'action': 'get_tweets', 'reject_url': tweet_url})
        paths = self.save_image_grid(images)
        self.tweets = tweets
        return tweets, paths
    