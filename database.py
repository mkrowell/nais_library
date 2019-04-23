#!/usr/bin/env python
'''
.. module::
    :language: Python Version 3.6.8
    :platform: Windows 10
    :synopsis: load NAIS data from MarineCadastre into a postgres database
    into a raw points table and a tracks table.

.. moduleauthor:: Maura Rowell <mkrowell@uw.edu>
'''


# ------------------------------------------------------------------------------
# IMPORTS
# ------------------------------------------------------------------------------
import datetime
from glob import glob
import numpy as np
import os
from os.path import exists, join
import pandas as pd
from postgis.psycopg import register
import psycopg2
from retrying import retry
import shutil
import tempfile
import webbrowser

from . import download_url, extract_zip, find_file


# ------------------------------------------------------------------------------
# MAIN CLASS
# ------------------------------------------------------------------------------
class NAIS_Database(object):

    '''
    Build PostgreSQL database of NAIS point data.
    '''

    def __init__(self, zone, year, password):
        # file parameters
        self.root = tempfile.mkdtemp()
        self.year = year
        self.zone = zone

        # spatial parameters
        self.lon_min = -125.98333333
        self.lon_max = -122.00000000
        self.lat_min = 45.46666667
        self.lat_max = 48.98333333

        # vessel parameters
        self.min_speed = 2

        # table parameters
        self.password = password
        self.columns = """
            MMSI char(9) CHECK (char_length(MMSI) = 9) NOT NULL,
            BaseDateTime timestamp NOT NULL,
            LAT float8 NOT NULL,
            LON float8 NOT NULL,
            SOG float(4) NOT NULL,
            COG float(4) NOT NULL,
            Heading float(4) NOT NULL,
            VesselName varchar(32),
            VesselType integer NOT NULL default -1,
            Status varchar(64),
            Length float(4),
            Width float(4),
            Draft float(4),
            Cargo integer default -1,
            TrackID integer NOT NULL
        """

        lines = [line.lstrip() for line in self.columns.split(",")]
        self.df_cols = [cl.split(' ')[0] for cl in lines]
        # TrackID is added in preprocessing
        self.df_cols.remove('TrackID')

        self.df_cols_not_null = [
            cl.split(' ')[0]
            for cl in lines
            if 'NOT NULL' in cl
        ]
        self.df_cols_not_null.remove('TrackID')

    @retry
    def download_raw(self):
        '''Dowload each month's raw data to a temp directory.'''
        for month in [str(i).zfill(2) for i in range(1, 13)]:
            raw = NAIS_Download(self.root, self.zone, month, self.year)
            raw.download_nais()

    def validate_boundary(self, df):
        '''Limit to boundary box.'''
        df = df[df['LON'].between(
            self.lon_min,
            self.lon_max,
            inclusive=True
        )].copy()
        df = df[df['LAT'].between(
            self.lat_min,
            self.lat_max,
            inclusive=True
        )].copy()
        return df

    def validate_speed(self, df):
        '''Drop rows with speeds under 2.'''
        return df[df['SOG'] > self.min_speed].copy()

    def validate_MMSI(self, df):
        '''Limit to MMSI of correct length.'''
        df['MMSI_Count'] = df['MMSI'].apply(lambda x: len(str(x)))
        df = df[df['MMSI_Count'] == 9].copy()
        return df.drop(columns=['MMSI_Count'])

    def validate_required(self, df):
        '''Handle NOT NULL columns.'''
        for col in self.df_cols_not_null:
            df[col].replace("", np.nan, inplace = True)
        return df.dropna(how='any', subset=self.df_cols_not_null)

    def add_track_id(self, df):
        '''Add track ID for each MMSI.'''
        df.sort_values(by=['MMSI', 'BaseDateTime'], inplace=True)
        df['TimeDiff'] = df.groupby('MMSI')['BaseDateTime'].diff()
        df['Break'] = np.where(df['TimeDiff'] > datetime.timedelta(minutes=6), 1, 0)
        df['TrackID'] = df.groupby('MMSI')['Break'].cumsum()
        return df.drop(columns=['TimeDiff', 'Break'])

    def validate_types(self, df):
        '''Cast to correct data types.'''
        cols_null = list(set(self.df_cols) - set(self.df_cols_not_null))
        for col in cols_null:
            df[col].fillna(-1, inplace=True)

        # Handle string columns
        df['Status'].replace(-1, "undefined", inplace=True)
        df['VesselName'].replace(-1, "undefined", inplace=True)

        # Handle int columns
        cols_int = ['VesselType', 'Cargo', 'TrackID']
        for col in cols_int:
            df[col] = df[col].astype(int)

        return df

    def preprocess_csv(self, csv_file):
        '''Clean real columns and write to temp dir.'''
        df = pd.read_csv(csv_file, usecols = self.df_cols)
        df['BaseDateTime'] = pd.to_datetime(df['BaseDateTime'])

        df = self.validate_boundary(df)
        df = self.validate_speed(df)
        df = self.validate_MMSI(df)
        df = self.validate_required(df)

        df = self.add_track_id(df)
        df = self.validate_types(df)

        # Save processed df to csv file
        df.to_csv(csv_file, index=False, header=False)

    def build_tables(self):
        '''Build database of raw data.'''
        self.download_raw()

        self.db = NAIS_Table(self.password, 'nais_points')
        self.db.drop_table()
        self.db.create_table(self.columns)

        # Copy data from temp directory and clean up
        for csv_file in glob(self.root + '\\*.csv'):
            print('Cleaning CSV file %s.' % csv_file)
            self.preprocess_csv(csv_file)
            print('Copying CSV file %s to database.' % csv_file)
            self.db.copy_data(csv_file, self.columns)
        shutil.rmtree(self.root)

        # Add PostGIS geometry and crete index for space and time
        self.db.add_geometry()
        self.db.add_index()

        # Create nais_track table with linestrings
        self.db.add_tracks()


# ------------------------------------------------------------------------------
# NAIS DOWNLOAD
# ------------------------------------------------------------------------------
class NAIS_Download(object):

    '''
    Download raw NAIS data from MarineCadastre to local temp directory.
    '''

    def __init__(self, root, zone, month, year):
        self.root = root
        self.year = year
        self.month = month
        self.zone = zone

        # Set name of downloaded file
        self.param_file = (self.year, self.month, self.zone)
        self.csv_raw = 'AIS_%s_%s_Zone%s.csv' % self.param_file

        # Download URL
        param = (self.year, self.year, self.month, self.zone)
        self.url = 'https://coast.noaa.gov/htdata/CMSP/AISDataHandler/%s/AIS_%s_%s_Zone%s.zip' % param

    def download_nais(self):
        '''Download zip file and extract to temp directory.'''
        # Prevent duplicate downloads
        if exists(join(self.root, self.csv_raw)):
            print('The NAIS download already exists for %s/%s Zone %s.' % self.param_file)
            return

        # Download file to temp directory, extract, and delete zip
        zfile = download_url(self.url, self.root, '.zip')
        extract_zip(zfile, self.root)
        os.remove(zfile)

        # Move extracted file to top level directory
        extracted_file = find_file(self.root, self.csv_raw)
        shutil.copy(extracted_file, self.root)

        # Remove subdirectories created during unzipping
        directory = join(self.root, 'AIS_ASCII_by_UTM_Month')
        if exists(directory):
            shutil.rmtree(directory)


# ------------------------------------------------------------------------------
# NAIS TABLE
# ------------------------------------------------------------------------------
class NAIS_Table():

    def __init__(self, password, table):
        '''Connect to default database.'''
        self.conn = psycopg2.connect(
            host='localhost',
            dbname='postgres',
            user='postgres',
            password=password)
        self.cur = self.conn.cursor()
        self.table = table

        self.cur.execute("CREATE EXTENSION IF NOT EXISTS postgis")
        self.conn.commit()
        register(self.conn)

    def set_timezone(self):
        '''Set timezone. The data is in UTC.'''
        self.cur.execute("SET timezone = 'UTC'")
        self.conn.commit()

    def drop_table(self):
        '''Drop the given table.'''
        self.cur.execute("DROP TABLE IF EXISTS {0}".format(self.table))
        self.conn.commit()

    def create_table(self, columns):
        '''Create given table.'''
        self.columns = columns
        self.cur.execute("CREATE TABLE {0} ({1})".format(self.table, self.columns))
        self.conn.commit()

    def copy_data(self, csv_file, headers):
        '''Copy data into table.'''
        with open(csv_file, 'r') as f:
            self.cur.copy_from(f, self.table, sep=',')
            self.conn.commit()

    def add_geometry(self):
        '''Add PostGIS Point geometry to the database and make it index.'''
        self.cur.execute("""
            ALTER TABLE {0}
            ADD COLUMN geom geometry(POINTM,4326)
            """.format(self.table)
        )
        print('Adding PostGIS POINTM to table.')
        self.cur.execute("""
            UPDATE {0}
            SET geom = ST_SetSRID(
                ST_MakePointM(lon,lat,date_part('epoch',basedatetime)),
                4326)
            """.format(self.table)
        )
        self.conn.commit()

    def add_index(self):
        print('Adding spatial index.')
        self.cur.execute("""
            CREATE INDEX idx_geom_box
            ON {0} USING GIST(geom)
            """.format(self.table)
        )
        print('Adding time index.')
        self.cur.execute("""
            CREATE INDEX idx_time
            ON {0} (basedatetime)
            """.format(self.table)
        )
        print('Adding MMSI index.')
        self.cur.execute("""
            CREATE INDEX idx_mmsi
            ON {0} (mmsi)
            """.format(self.table)
        )
        self.conn.commit()

    def add_tracks(self):
        '''Add LINESTRING for each MMSI, TrackID.'''
        self.cur.execute("DROP TABLE IF EXISTS nais_tracks")
        self.conn.commit()
        self.cur.execute("""
            CREATE TABLE nais_tracks AS
            SELECT mmsi, trackid, ST_MakeLine(geom ORDER BY basedatetime) AS track
            FROM {0}
            GROUP BY mmsi, trackid
            """.format(self.table)
        )
        self.conn.commit()
