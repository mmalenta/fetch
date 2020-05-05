#!/usr/bin/env python3

import argparse
import json
import logging
import pathlib
import pika

import numpy as np
from os import path
from pysigproc import SigprocFile
from candidate import *
from gpu_utils import gpu_dedisp_and_dmt_crop
from socket import gethostname
from time import time

from mtcutils import core as mtcore
from mtcutils import iqrm_mask

logger = logging.getLogger("candmaker")

class Candmaker:

    def __init__(self, config):
        self._verbose = config['verbose']
        self._opt_dm = config['opt_dm']
        self._time_size = config['time_size']
        self._freq_size = config['freq_size']
        self._cand_data = None
        self._cand_h5 = ""
        self._hostname = gethostname()


    def Candmake(self):

        connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
        channel = connection.channel()

        channel.exchange_declare(exchange='candidates_' + self._hostname,
                                    exchange_type='direct',
                                    durable=True)
        
        channel.queue_declare('fetch_' + self._hostname, durable=True)
        channel.queue_bind('fetch_' + self._hostname, 'candidates_' + self._hostname)

        channel.queue_declare('fetch_predictor_' + self._hostname, durable=True)
        channel.queue_bind('fetch_predictor_' + self._hostname, 'candidates_' + self._hostname)

        channel.basic_qos(prefetch_count=1)

        channel.basic_consume(queue='fetch_' + self._hostname, auto_ack=False, on_message_callback=self.Cand2H5)
        channel.start_consuming()

    def Cand2H5(self, ch, method, properties, body):

        self._cand_data = json.loads(body.decode('utf8'))
        # Extra variables
        base_dir = self._cand_data['base_dir']
        beam_dir = self._cand_data['beam_dir']
        # TODO: remember to add mask file to the published message
        mask_file = self._cand_data['mask_file']
        # Usual FETCH variables
        fil_name = self._cand_data['filterbank']
        snr = float(self._cand_data['cand_snr'])
        width = 2 ** float(self._cand_data['cand_width'])
        dm = float(self._cand_data['cand_dm'])
        tcand = float(self._cand_data['cand_start_time'])
        label = 1
        # TODO: remember to add beam information to the published message
        beam = self._cand_data['cand_beam']
        beam_type = self._cand_data['cand_beam_type']

        logger.info("Processing file % s", (fil_name))

        filobj = SigprocFile(path.join(base_dir, beam_dir, fil_name))
        logger.info("Creating 'mask' for %d channels", filobj.nchans)
        mask = np.zeros(filobj.nchans, dtype=np.bool)

        cand = Candidate(path.join(base_dir, beam_dir, fil_name), snr=snr, width=width, dm=dm, label=label, tcand=tcand, kill_mask=mask)
        cand.get_chunk()
        cand.fp.close()

        # NOTE: mean is currently dummy variables that contain per-channel mean and standard deviation of the original data.
        # Should we include that in the archive?
        scaled, mean, std = mtcore.normalise(cand.data.T)
        # NOTE: Per note above - mtcutils iqrm_mask() returns logical mask
        # Should we include iqrm mask in the archive?
        mask = iqrm_mask(std, maxlag=3)
        scaled[mask] = 0
        cand.data = scaled.T
        
        dedisp_start = time()

        cand=gpu_dedisp_and_dmt_crop(cand)
        cand.dedispersed=resize(cand.dedispersed, (256,256))

        cand.dmt = self.__NormaliseClip(cand.dmt)
        cand.dedispersed = self.__NormaliseClip(cand.dedispersed, 3)
        
        dedisp_end = time()
        logger.info("Dedispersion took %.2f seconds", (dedisp_end - dedisp_start))

        fmtdm = "{:.2f}".format(dm)
        self._cand_h5 = path.join(beam_dir, str(tcand) + '_DM_' + fmtdm + '_beam_' + str(beam) + beam_type + fil_name +  '.hdf5')
        
        print(self._cand_h5)

        fout = cand.save_h5(self._cand_h5)
        if self._verbose:
            print("Created file %s" % (self._cand_h5))

        pred_json = {
            'beam_dir': beam_dir,
            'filterbank': fil_name,
            'candidate_h5': self._cand_h5
        }

        pika_properties = pika.BasicProperties(content_type='application/json',
                                                content_encoding='utf-8',
                                                delivery_mode=2)

        # TODO: Change it to self._hostname
        self._hostname = gethostname()

        ch.basic_ack(delivery_tag=method.delivery_tag)
        ch.basic_publish(exchange='candidates_' + self._hostname,
                                routing_key='fetch_predictor_' + self._hostname,
                                body=json.dumps(pred_json),
                                properties=pika_properties)


    def __NormaliseClip(self, data, clip_range=None):
        """
        Noramlise the data by unit standard deviation and zero median.
        Clip the resulting data to withing 3 sigma.
        :param data: data
        :return:
        """
        data = np.array(data, dtype=np.float32)
        median = np.median(data)
        std = np.std(data)
        logging.debug(f'Data median: {median}')
        logging.debug(f'Data std: {std}')
        data -= median
        data /= std

        if clip_range != None:
            data = np.clip(data, -1.0 * clip_range, clip_range) 
        return data


def cand2h5(cand_val):
    """
    TODO: Add option to use cand.resize for reshaping FT and DMT
    Generates h5 file of candidate with resized frequency-time and DM-time arrays
    :param cand_val: List of candidate parameters (fil_name, snr, width, dm, label, tcand(s))
    :type cand_val: Candidate
    :return: None
    """
    fil_name, snr, width, dm, label, tcand, kill_mask_path, args = cand_val
    if kill_mask_path == kill_mask_path:
        kill_mask_file = pathlib.Path(kill_mask_path)
        if kill_mask_file.is_file():
            logging.info(f'Using mask {kill_mask_path}')
            kill_chans = np.loadtxt(kill_mask_path, dtype=np.int)
            filobj = SigprocFile(fil_name)
            kill_mask = np.zeros(filobj.nchans, dtype=np.bool)
            kill_mask[kill_chans]= True

    else:
        logging.debug('No Kill Mask')
        kill_mask = None

    cand = Candidate(fil_name, snr=snr, width=width, dm=dm, label=label, tcand=tcand, kill_mask=kill_mask)
    cand.get_chunk()
    cand.fp.close()
    logging.info('Got Chunk')
    cand.dmtime()
    logging.info('Made DMT')
    if args.opt_dm:
        logging.info('Optimising DM')
        logging.warning('This feature is experimental!')
        cand.optimize_dm()
    else:
        cand.dm_opt = -1
        cand.snr_opt = -1
    cand.dedisperse()
    logging.info('Made Dedispersed profile')

    pulse_width = cand.width
    if pulse_width == 1:
        time_decimate_factor = 1
    else:
        time_decimate_factor = pulse_width // 2

    # Frequency - Time reshaping
    cand.decimate(key='ft', axis=0, pad=True, decimate_factor=time_decimate_factor, mode='median')
    crop_start_sample_ft = cand.dedispersed.shape[0] // 2 - args.time_size // 2
    cand.dedispersed = crop(cand.dedispersed, crop_start_sample_ft, args.time_size, 0)
    logging.info(f'Decimated Time axis of FT to tsize: {cand.dedispersed.shape[0]}')

    if cand.dedispersed.shape[1] % args.frequency_size == 0:
        cand.decimate(key='ft', axis=1, pad=True, decimate_factor=cand.dedispersed.shape[1] // args.frequency_size,
                      mode='median')
        logging.info(f'Decimated Frequency axis of FT to fsize: {cand.dedispersed.shape[1]}')
    else:
        cand.resize(key='ft', size=args.frequency_size, axis=1, anti_aliasing=True)
        logging.info(f'Resized Frequency axis of FT to fsize: {cand.dedispersed.shape[1]}')

    # DM-time reshaping
    cand.decimate(key='dmt', axis=1, pad=True, decimate_factor=time_decimate_factor, mode='median')
    crop_start_sample_dmt = cand.dmt.shape[1] // 2 - args.time_size // 2
    cand.dmt = crop(cand.dmt, crop_start_sample_dmt, args.time_size, 1)
    logging.info(f'Decimated DM-Time to dmsize: {cand.dmt.shape[0]} and tsize: {cand.dmt.shape[1]}')

    cand.dmt = normalise(cand.dmt)
    cand.dedispersed = normalise(cand.dedispersed)

    fout = cand.save_h5(fnout=self._cand_h5)
    logging.info(fout)
    if args.plot:
        logging.info('Displaying the candidate')
        plot_h5(fout, show=False, save=True, detrend=False)
    return None

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', help='Be verbose', action='store_true')
    parser.add_argument('-fs', '--freq_size', type=int, help='Frequency size after rebinning', default=256)
    parser.add_argument('-ts', '--time_size', type=int, help='Time length after rebinning', default=256)
    parser.add_argument('-opt', '--opt_dm', dest='opt_dm', help='Optimise DM', action='store_true', default=False)
    arguments = parser.parse_args()

    configuration = {
        'verbose': arguments.verbose,
        'opt_dm': arguments.opt_dm,
        'time_size': arguments.time_size,
        'freq_size': arguments.freq_size,
    }

    logging_format = '%(asctime)s - %(funcName)s -%(name)s - %(levelname)s - %(message)s'

    if arguments.verbose:
        logging.basicConfig(level=logging.DEBUG, format=logging_format)
    else:
        logging.basicConfig(level=logging.INFO, format=logging_format)

    candmaker = Candmaker(configuration)
    candmaker.Candmake()


if __name__ == '__main__':
    main()

