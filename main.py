import os
import pandas as pd
import numpy as np
import shutil
import logging
import scipy.stats as stats
import warnings
import ast
import re
import json
import time
import torch
import gc

from tqdm import tqdm
from lightfm.data import Dataset
from lightfm import LightFM
from logging import getLogger
from typing import List, Tuple
from collections import defaultdict
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.sequential_recommender import GRU4Rec
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger
from recbole.utils.case_study import full_sort_topk


DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# функция для генерации LightFM предсказаний по всем пользователям с учетом удаления просмотренных айтемов
def generate_lightfm_recs_mapper(model, item_ids, known_items,
                                 user_features, item_features, N,
                                 user_mapping, item_inv_mapping,
                                 num_threads=1):
    def _recs_mapper(user):
        user_id = user_mapping[user]
        recs = model.predict(user_id, item_ids, user_features=user_features,
                             item_features=item_features, num_threads=num_threads)

        additional_N = len(known_items[user_id]) if user_id in known_items else 0
        total_N = N + additional_N
        top_cols = np.argpartition(recs, -np.arange(total_N))[-total_N:][::-1]

        final_recs = [item_inv_mapping[item] for item in top_cols]
        if additional_N > 0:
            filter_items = known_items[user_id]
            final_recs = [item for item in final_recs if item not in filter_items]
        return final_recs[:N]
    return _recs_mapper

# Функция, которая проверяет, является ли значение не списком,
# и возвращает список топ 20 фильмов или исходное значение
def fill_with_top_20(value):
    if not isinstance(value, list): 
        return top_20 
    else: 
        return value


if __name__ == '__main__':
    countries = pd.read_csv('/content/drive/MyDrive/train/countries.csv')
    countries['name'] = countries['name'].str.lower()
    countries['name'] = countries['name'].str.strip().replace(' ', '')

    genres = pd.read_csv('/content/drive/MyDrive/train/genres.csv')
    genres['name'] = genres['name'].str.lower()
    genres['name'] = genres['name'].str.strip()#.replace(' ', '')

    staff = pd.read_csv('/content/drive/MyDrive/train/staff.csv')
    staff['name'] = staff['name'].str.strip()
    staff['name'] = staff['name'].str.replace(' ', '')

    movies = pd.read_csv('/content/drive/MyDrive/train/movies.csv', parse_dates=['year', 'date_publication'])
    movies['year'] = movies['year'].dt.year
    movies.rename(columns={"id": "movie_id"}, inplace=True)

    log = pd.read_csv('/content/drive/MyDrive/train/logs.csv', parse_dates=['datetime'])
    logs = log.copy()
    logs = logs.sort_values('datetime')
    logs['movie_id'] = logs['movie_id'].astype(int)
    logs['duration'] = logs['duration'].astype(int)
    logs['datetime'] = logs['datetime'].astype('datetime64[ns]')
    logs['timestamp'] = pd.to_datetime(logs['datetime']).astype(int) // 10**9

    # Заменим выброс предыдущим значением duration для этого фильма
    logs.loc[logs['movie_id'] == 6442, 'duration'] = \
    logs[logs['movie_id'] == 6442]['duration'].sort_values(ascending=False).values[1]
    
    # Удалим рекламу и т.п.
    movies.drop(movies[movies['name'].str.contains('test')].index, inplace=True)
    movies.drop(movies[movies['name'].str.contains('WM')].index, inplace=True)
    movies.drop(movies[movies['name'].str.contains(r'[A-Z]{6}')].index, inplace=True)
    movies.drop(movies[movies['name'].str.contains(r'скидк.*%.*', case=False)].index, inplace=True)
    movies.drop(movies[movies['name'].str.contains(r'триколор', case=False)].index, inplace=True)
    # удалим лишнее
    # movies['description'] = movies['description'].str.strip()
    # movies['description'] = movies['description'].str.replace("ДОСТУПНО В ПОДПИСКЕ AMEDIATEKA\n", "")
    #
    movies.drop(movies[~movies['description'].isna() & movies['description'].str.contains(r'промокод', case=False)].index, inplace=True)
    movies.drop(movies[~movies['description'].isna() & movies['description'].str.contains(r'подписк', case=False)].index, inplace=True)

    movies = movies[(movies['year'] > 1900) & (movies['year'] < 2024)]
    #
    # movies.loc[movies['movie_id'] == 3420, 'year'] = 2014
    # movies.loc[movies['movie_id'] == 4962, 'year'] = 2014


    # Дубли фильмов
    logs.loc[logs['movie_id'] == 535, 'movie_id'] = 5014
    logs.loc[logs['movie_id'] == 5808, 'movie_id'] = 2679

    movie_id_duplicates = [81,492,535,
                        1012,1024,1292,1921,1969,
                        2201,2307,2625,2706,
                        3182,3647,3922,
                        4016,4031,4106,4228,4371,4439,4822,
                        5443,5710,5730,5808,5994,
                        6163,6193,6401,6801,6956,
                        7260,7297,7358]
    movies = movies[~movies['movie_id'].isin(movie_id_duplicates)]

    logs.drop(logs[~logs['movie_id'].isin(movies['movie_id'])].index, inplace=True)

    logs.drop_duplicates(subset=['user_id', 'movie_id'], keep='last', inplace=True)

    del countries
    del genres
    del staff


    dataset = Dataset()
    dataset.fit(logs['user_id'].unique(), logs['movie_id'].unique())

    interactions_matrix, weights_matrix = dataset.build_interactions(
        zip(*logs[['user_id', 'movie_id', 'duration']].values.T))

    weights_matrix_csr = weights_matrix.tocsr()

    lightfm_mapping = dataset.mapping()
    lightfm_mapping = {'users_mapping': lightfm_mapping[0], 'items_mapping': lightfm_mapping[2]}

    lightfm_mapping['users_inv_mapping'] = {v: k for k, v in lightfm_mapping['users_mapping'].items()}
    lightfm_mapping['items_inv_mapping'] = {v: k for k, v in lightfm_mapping['items_mapping'].items()}

    lfm_params = {
        'no_components': 64,
        'learning_rate': 0.01,
        'loss': 'warp',
        'max_sampled': 5,
        'random_state': 42
    }
    lfm_model = LightFM(**lfm_params)

    num_epochs = 50
    for _ in tqdm(range(num_epochs)):
        lfm_model.fit_partial(weights_matrix_csr)

    # кол-во кандидатов
    top_N = 20
    # вспомогательные данные
    all_cols = list(lightfm_mapping['items_mapping'].values())

    mapper = generate_lightfm_recs_mapper(
        lfm_model,
        item_ids=all_cols,
        known_items=dict(),
        N=top_N,
        user_features=None,
        item_features=None,
        user_mapping=lightfm_mapping['users_mapping'],
        item_inv_mapping=lightfm_mapping['items_inv_mapping'],
        num_threads=20
    )

    # генерируем предказания
    lfm_candidates = pd.DataFrame({'user_id': logs['user_id'].unique()})
    lfm_candidates['movie_id'] = lfm_candidates['user_id'].map(mapper)


    lfm_candidates = lfm_candidates.explode('movie_id')
    lfm_candidates['user_id'] = lfm_candidates['user_id'].astype(int)
    lfm_candidates['movie_id'] = lfm_candidates['movie_id'].astype(int)
    lfm_candidates = lfm_candidates.groupby('user_id')['movie_id'].agg(list).reset_index()

    # чистим кэш и мусор
    torch.cuda.empty_cache()
    gc.collect()


    # Сохраняем данные logs в фомате recbole
    recbole_data_inter = logs[['user_id', 'movie_id', 'timestamp', 'duration']]
    recbole_data_inter.columns = ['user_id:token', 'item_id:token', 'timestamp:float', 'duration:float']

    recbole_folder = '/recbole_data'
    if not os.path.exists(recbole_folder):
        os.makedirs(recbole_folder)
    recbole_data_inter_folder = ('/recbole_data/' + 'recbole_data.inter')
    # if not os.path.exists(recbole_data_inter_folder):
    recbole_data_inter.to_csv(recbole_data_inter_folder, sep='\t', index=False)

    # Инициализируем и обучаем модель GRU4Rec
    config_dict = {
        "USER_ID_FIELD": "user_id",
        "ITEM_ID_FIELD": "item_id",
        "TIME_FIELD": "timestamp",
        "load_col": {"inter": ["user_id", "item_id", "timestamp", "duration"]},
        "ITEM_LIST_LENGTH_FIELD": "item_length",
        "LIST_SUFFIX": "_list",
        "MAX_ITEM_LIST_LENGTH": 130,
        "embedding_size": 256,
        "hidden_size": 256,
        "num_layers": 2,
        "dropout_prob": 0.3,
        "loss_type": "CE",
        "epochs": 50,
        "train_batch_size": 1024,
        "eval_batch_size": 1024,
        "train_neg_sample_args": None,
        "eval_args": {
            "group_by": "user",
            "order": "TO",
            "split": {"LS": "valid_only"},
            "mode": "full",
        },
        "metrics": "MAP",
        "topk": 20,
        "valid_metric": "MAP@20",
        "data_path": "/",
        "stopping_step": 2,
        "device": DEVICE,
    }
    config = Config(model='GRU4Rec', dataset='recbole_data', config_dict=config_dict)

    init_seed(config['seed'], config['reproducibility'])

    logger = getLogger()
    logger.setLevel(logging.INFO)
    c_handler = logging.StreamHandler()
    c_handler.setLevel(logging.INFO)
    logger.addHandler(c_handler)
    logger.info(config)

    dataset = create_dataset(config)
    logger.info(dataset)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    g4r_model = GRU4Rec(config, train_data.dataset).to(config['device'])
    logger.info(g4r_model)
    trainer = Trainer(config, g4r_model)
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)

    # Делаем предсказания valid_data
    topk_items = []
    for internal_user_id in list(range(dataset.user_num))[1:]:
        _, topk_iid_list = full_sort_topk([internal_user_id],
                                          g4r_model,
                                          valid_data,
                                          k=20,
                                          device=config['device'])
        if topk_iid_list.shape[0] > 0:
            last_topk_iid_list = topk_iid_list[-1]
            external_item_list = dataset.id2token(dataset.iid_field, last_topk_iid_list.cpu()).tolist()
            topk_items.append(external_item_list)

    external_user_ids = dataset.id2token(dataset.uid_field, list(range(dataset.user_num)))[1:]

    external_item_str = [' '.join(x) for x in topk_items]

    g4r_candidates = pd.DataFrame(external_user_ids, columns=['user_id'])
    if len(external_item_str) != len(g4r_candidates):
        external_item_str.extend([np.nan] * (len(g4r_candidates) - len(external_item_str)))

    g4r_candidates['movie_id'] = external_item_str
    g4r_candidates.dropna(inplace=True)

    g4r_candidates['movie_id'] = g4r_candidates['movie_id'].str.split(' ').map(list)

    g4r_candidates['movie_id'] = g4r_candidates['movie_id'].apply(lambda x: ast.literal_eval(x))
    g4r_candidates = g4r_candidates.explode('movie_id')
    g4r_candidates['user_id'] = g4r_candidates['user_id'].astype(int)
    g4r_candidates['movie_id'] = g4r_candidates['movie_id'].astype(int)
    g4r_candidates = g4r_candidates.groupby('user_id')['movie_id'].agg(list).reset_index()


    # чистим кэш и мусор
    torch.cuda.empty_cache()
    gc.collect()


    # Создаём итоговый датафрейм рекомендаций
    predict = pd.DataFrame({'user_id': log['user_id'].unique()})

    # Заполняем рекомендациями LightFM
    predict = predict.merge(lfm_candidates, on='user_id', how='left')

    # Обновляем рекомендациями GRU4Rec
    predict.set_index('user_id')
    g4r_candidates.set_index('user_id')
    predict.update(g4r_candidates)
    predict.reset_index()

    # Получите список топ 20 фильмов из датафрейма logs
    top_20 = logs['movie_id'].value_counts().head(20).index.to_list()
    # Примените функцию к колонке movie_id датафрейма predict
    predict['movie_id'] = predict['movie_id'].apply(fill_with_top_20)


    predict.info()
    predict['user_id'] = predict['user_id'].astype(int)

    # Сохраняем итоговый датафрейм рекомендаций
    os.mkdir('output')
    predict.to_csv('output/result.csv', index=False, header=False)