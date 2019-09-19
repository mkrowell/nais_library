#!/usr/bin/env python
'''
.. module::
    :language: Python Version 3.6.8
    :platform: Windows 10
    :synopsis: download shoreline, TSS, and NAIS data

.. moduleauthor:: Maura Rowell <mkrowell@uw.edu>
'''


# ------------------------------------------------------------------------------
# IMPORTS
# ------------------------------------------------------------------------------
from glob import glob
import io
from multiprocessing.dummy import Pool
import os
from os.path import dirname, exists, join
import requests
from retrying import retry
import shutil
import zipfile
import yaml

from . import download_url, extract_zip, find_file
from . import dataframes


# ------------------------------------------------------------------------------
# DOWNLOADS
# ------------------------------------------------------------------------------
class Shoreline_Download(object):

    '''
    Download a GIS representation of the United States shoreline
    and save it to the directory provided.
    '''

    def __init__(self, root):
        self.root = root
        self.url = 'https://coast.noaa.gov/htdata/Shoreline/us_medium_shoreline.zip'

    def download_shoreline(self):
        '''Download zip file and extract to temp directory.'''
        output = join(self.root, 'us_medium_shoreline.shp')
        if exists(output):
            print('The Shoreline shapefile has already been downloaded.')
            return output

        print('Downloading the Shoreline shapefile...')
        zfile = download_url(self.url, self.root, '.zip')
        extract_zip(zfile, self.root)
        os.remove(zfile)
        return output

class TSS_Download(object):

    '''
    Download a GIS representation of the US Traffic Separation Scheme
    and save it to the directory provided.
    '''

    def __init__(self, root):
        self.root = root
        self.url = 'http://encdirect.noaa.gov/theme_layers/data/shipping_lanes/shippinglanes.zip'

    def download_tss(self):
        '''Download zip file and extract to temp directory.'''
        output = join(self.root, 'shippinglanes.shp')
        if exists(output):
            print('The TSS shapefile has already been downloaded.')
            return output

        print('Downloading the TSS shapefile...')
        download = requests.get(self.url)
        zfile = zipfile.ZipFile(io.BytesIO(download.content))
        zfile.extractall(self.root)
        return output

class NAIS_Download(object):

    '''
    Download raw NAIS data from MarineCadastre for the given city and year
    and save it to the directory provided.
    '''

    def __init__(self, root, city, year):
        self.root = root
        self.year = year
        self.city = city

        param_yaml = join(dirname(__file__), 'settings.yaml')
        with open(param_yaml, 'r') as stream:
            self.parameters = yaml.safe_load(stream)[self.city]

        self.zone = self.parameters['zone']
        self.lonMin = self.parameters['lonMin']
        self.lonMax = self.parameters['lonMax']
        self.latMin = self.parameters['latMin']
        self.latMax = self.parameters['latMax']
        self.stepSize = self.parameters['stepSize']

        self.name = 'AIS_{0}_{1}_Zone{2}.csv'
        self.url = 'https://coast.noaa.gov/htdata/CMSP/AISDataHandler/{0}/AIS_{1}_{2}_Zone{3}.zip'
        self.download_dir = join(self.root, 'AIS_ASCII_by_UTM_Month')

        self.df = None

    @retry(stop_max_attempt_number=7)
    def download_nais(self, month):
        '''Download zip file and extract to temp directory.'''
        name = self.name.format(self.year, month, self.zone)
        csv = join(self.root, name)
        if exists(csv):
            return

        print('Downloading NAIS file for month {0}...'.format(month))
        url = self.url.format(self.year, self.year, month, self.zone)
        zfile = download_url(url, self.root, '.zip')
        extract_zip(zfile, self.root)

        # Move to top level directory
        extracted_file = find_file(self.root, name)
        shutil.copy(extracted_file, self.root)

    def clean_up(self):
        '''Remove subdirectories created during unzipping.'''
        if exists(self.download_dir):
            shutil.rmtree(self.download_dir)

    def preprocess_eda(self):
        '''Add derived fields and validate data types.'''
        csvs = glob(self.root + '\\AIS*.csv')
        self.df = dataframes.EDA_Dataframe(
            csvs,
            self.lonMin,
            self.lonMax,
            self.latMin,
            self.latMax
        )

        self.df.clean_raw()
        self.df.mark_enter_exit()
        self.df.step_time()
        self.df.mark_jump()
        self.df.reorder_output()
        
    def preprocess_nais(self):
        '''Add derived fields and validate data types.'''
        csvs = glob(self.root + '\\AIS*.csv')
        self.df = dataframes.NAIS_Dataframe(
            csvs,
            self.lonMin,
            self.lonMax,
            self.latMin,
            self.latMax
        )

        self.df.simplify()
        self.df.clean()
        self.df.split_mmsi_stop()
        self.df.add_evasive_data()
        self.df.save_output()
