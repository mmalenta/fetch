#!/usr/bin/env python3

import argparse
import glob
import logging
import os
import string

import numpy as np
import pandas as pd
import pika
import json

from socket import gethostname
from fetch.data_sequence import DataGenerator
from fetch.utils import get_model
from string import ascii_lowercase
from numba import cuda

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

class Predictor:

    def __init__(self, config):

        tf_config = tf.ConfigProto()
        tf_config.gpu_options.per_process_gpu_memory_fraction = 0.25
        set_session(tf.Session(config=tf_config))

        cuda.select_device(0)

        self._verbose = config['verbose']
        self._threshold = config['threshold']

        if config['model'] not in list(ascii_lowercase)[:11]:
            raise ValueError('Model only range from a -- j.')
        else:
            self._model = get_model(config['model'])

        if config['gpu'] >= 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(config['gpu'])
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

    def Predict(self):

        hostname = gethostname()

        connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
        channel = connection.channel()

        channel.exchange_declare(exchange='candidates_' + hostname,
                                    exchange_type='direct',
                                    durable=True)

        channel.queue_declare('fetch_predictor_' + hostname, durable=True)
        channel.queue_bind('fetch_predictor_' + hostname, 'candidates_' + hostname)

        channel.basic_qos(prefetch_count=1)

        channel.basic_consume(queue='fetch_predictor_' + hostname, auto_ack=False, on_message_callback=self.GetPrediction)
        channel.start_consuming()

    def GetPrediction(self, ch, method, properties, body):
        
        cand_data = json.loads(body.decode('utf8'))
        cand_h5 = cand_data['candidate_h5']

        if self._verbose:
            print("Running predictor for file %s..." % (cand_h5))

        cand_datagen = DataGenerator(list_IDs = [cand_h5], labels=[0], shuffle=False, noise=False, batch_size=1)

        cand_probs = self._model.predict_generator(generator=cand_datagen, max_queue_size=1, verbose=1, use_multiprocessing=False,
                                    workers=1, steps=len(cand_datagen))

        hostname = gethostname()

        for idx, cand in enumerate(cand_probs):

            print(idx, cand)
            results_dict = {}
            results_dict['beam_dir'] = cand_data['beam_dir']
            results_dict['filterbank'] = cand_data['filterbank']
            results_dict['candidate'] = cand_data['candidate_h5']
            results_dict['probability'] = cand[1].item(0)
            results_dict['label'] = int(np.round(cand[1].item(0) >= self._threshold))
            print(results_dict)
        
            ch.basic_publish(exchange='candidates_' + hostname,
                            routing_key='fetch',
                            body=json.dumps(results_dict))

        ch.basic_ack(delivery_tag=method.delivery_tag)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast Extragalactic Transient Candiate Hunter (FETCH)",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', help='Be verbose', action='store_true')
    parser.add_argument('-g', '--gpu_id', help='GPU ID (use -1 for CPU)', type=int, required=False, default=0)
    parser.add_argument('-m', '--model', help='Index of the model to train', required=True)
    parser.add_argument('-t', '--threshold', help='Detection threshold', default=0.5, type=float)
    arguments = parser.parse_args()

    configuration = {
        'verbose': arguments.verbose,
        'gpu': arguments.gpu_id,
        'model': arguments.model,
        'threshold': arguments.threshold
    }

    predictor = Predictor(configuration)
    predictor.Predict()