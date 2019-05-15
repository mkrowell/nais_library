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
import io
import math
import matplotlib.pyplot as plt
from multiprocessing.dummy import Pool
import numpy as np
import os
from os.path import dirname, exists, join
import osgeo.ogr
import pandas as pd
from postgis.psycopg import register
import psycopg2
from psycopg2 import sql
import requests
from retrying import retry
import shutil
import subprocess
import tempfile
import time
import webbrowser
import zipfile
import yaml

from . import download_url, extract_zip, find_file


# ------------------------------------------------------------------------------
# MAIN CLASS
# ------------------------------------------------------------------------------
class NAIS_Database(object):

    '''
    Build PostgreSQL database of NAIS point data.
    '''

    def __init__(self, city, year, password):
        # arguments
        self.city = city
        self.year = year
        self.password = password

        # spatial parameters
        param_yaml = join(dirname(__file__), 'settings.yaml')
        with open(param_yaml, 'r') as stream:
            self.parameters = yaml.safe_load(stream)[self.city]

        # self.parameters = yaml.load(open(parameters_yaml).read())
        self.zone = self.parameters['zone']
        self.lonMin = self.parameters['lonMin']
        self.lonMax = self.parameters['lonMax']
        self.latMin = self.parameters['latMin']
        self.latMax = self.parameters['latMax']
        self.stepSize = self.parameters['stepSize']

        # time parameters
        self.months = [str(i).zfill(2) for i in range(1, 13)]

        # database parameters
        self.conn = psycopg2.connect(
            host='localhost',
            dbname='postgres',
            user='postgres',
            password=self.password)

        # tables
        self.table_shore = Shapefile_Table(self.conn, 'shore')
        self.table_tss = Shapefile_Table(self.conn, 'tss')

        self.table_points = Points_Table(
            self.conn,
            'nais_points_{0}'.format(self.zone)
        )
        self.table_near = Near_Table(
            self.conn,
            'near_points_{0}'.format(self.zone),
            self.table_points.table
        )

        self.table_tracks = Tracks_Table(
            self.conn,
            'nais_tracks_{0}'.format(self.zone)
        )
        self.table_cpa = CPA_Table(
            self.conn,
            'near_tracks_{0}'.format(self.zone),
            self.table_tracks.table
        )

        self.table_grid = Grid_Table(self.conn, 'grid_{0}'.format(self.zone))

        # file parameters
        self.root = tempfile.mkdtemp()

    @property
    def nais_csvs(self):
        return glob(self.root + '\\AIS*.csv')

    # --------------------------------------------------------------------------
    # MAIN METHOD
    # --------------------------------------------------------------------------
    def build_tables(self):
        '''Build database of raw data.'''
        start = time.time()
        try:
            self.build_shore()
            self.build_tss()
            self.build_grid(overwrite=False)
            
            self.build_nais_points(overwrite=False)
            self.build_nais_tracks(overwrite=False)
            
            # Make near points table
            self.table_near.drop_table()
            self.table_near.near_points()

            # Make near tracks table
            self.table_cpa.drop_table()
            self.table_cpa.near_tracks()

        except Exception as err:
            print(err)
            self.conn.rollback()
        finally:
            shutil.rmtree(self.root)
            end = time.time()
            print('Elapsed Time: {0} minutes'.format((end-start)/60))

    # --------------------------------------------------------------------------
    # UTILITY METHODS
    # --------------------------------------------------------------------------
    def build_shore(self):
        '''Construct shoreline table.'''
        print('Constructing shoreline table...')
        shore = Shoreline_Download(self.root)
        self.shoreline_shp = shore.download_shoreline()
        self.table_shore.create_table(filepath=self.shoreline_shp)
        self.table_shore.reduce_table(self.parameters['region'])

    def build_tss(self):
        '''Construct TSS table.'''
        print('Constructing tss table...')
        tss = TSS_Download(self.root)
        self.tss_shp = tss.download_tss()
        self.table_tss.create_table(filepath=self.tss_shp)

    def build_grid(self):
        '''Create grid table.'''
        print('Constructing grid table...')
        self.grid_df = Sector_Dataframe(
            self.lonMin,
            self.lonMax,
            self.latMin,
            self.latMax,
            self.stepSize
        ).generate_df()
        self.grid_csv = join(self.root, 'grid_table.csv')
        self.grid_df.to_csv(self.grid_csv, index=True, header=False)
        self.table_grid.drop_table()

        self.table_grid.create_table()
        self.table_grid.copy_data(self.grid_csv)
        self.table_grid.add_points()
        # self.table_grid.make_bounding_box()

    def download_raw(self):
        '''Dowload raw data to a temp directory.'''
        raw = NAIS_Download(self.root, self.city, self.year)
        for month in self.months:
            raw.download_nais(month)

    def preprocess_raw(self):
        '''Process the raw csv files.'''
        raw = NAIS_Download(self.root, self.city, self.year)
        with Pool(processes=12) as pool:
            pool.map(raw.preprocess_nais, self.months)

    def build_nais_points(self):
        '''Build nais points table.'''
        print('Constructing nais_points table...')
        self.download_raw()
        self.preprocess_raw()
        self.table_points.drop_table()
        self.table_points.create_table()
        self.table_points.set_parallel(17)
        self.table_points.set_timezone()

        for csv_file in self.nais_csvs:
            self.table_points.copy_data(csv_file)

        self.table_points.add_geometry()
        self.table_points.add_tss()
        self.table_points.add_indexes()

    def build_nais_tracks(self):
        '''Create tracks table from points table.'''
        print('Constructing nais_tracks table...')
        self.table_tracks.drop_table()
        self.table_tracks.add_tracks()

        print('Removing tracks that cross shore...')
        self.table_tracks.reduce_table()


# ------------------------------------------------------------------------------
# DOWNLOADS
# ------------------------------------------------------------------------------
class Shoreline_Download(object):

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
    Download raw NAIS data from MarineCadastre to local temp directory.
    '''

    def __init__(self, root, city, year):
        self.root = root
        self.year = year
        self.city = city

        param_yaml = join(dirname(__file__), 'settings.yaml')
        with open(param_yaml, 'r') as stream:
            self.parameters = yaml.safe_load(stream)[self.city]

        # self.parameters = yaml.load(open(parameters_yaml).read())
        self.zone = self.parameters['zone']
        self.lonMin = self.parameters['lonMin']
        self.lonMax = self.parameters['lonMax']
        self.latMin = self.parameters['latMin']
        self.latMax = self.parameters['latMax']
        self.stepSize = self.parameters['stepSize']

        self.name = 'AIS_{0}_{1}_Zone{2}.csv'
        self.url = 'https://coast.noaa.gov/htdata/CMSP/AISDataHandler/{0}/AIS_{1}_{2}_Zone{3}.zip'
        self.download_dir = join(self.root, 'AIS_ASCII_by_UTM_Month')

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

    def preprocess_nais(self, month):
        '''Add derived fields and validate data types.'''
        csv = join(self.root, self.name.format(self.year, month, self.zone))
        print('Preprocessing file for month {0}...'.format(month))
        df = NAIS_Dataframe(
            csv,
            self.lonMin,
            self.lonMax,
            self.latMin,
            self.latMax,
            self.stepSize
        )

        # Reduce data
        df.drop_bad_data()
        df.drop_moored()
        df.drop_spatial()

        # Add derived fields
        df.add_sector()
        df.add_track()
        df.add_ROT()
        df.add_interval()

        # Handle data types
        df.validate_types()

        # Write
        df.reorder_output()


# ------------------------------------------------------------------------------
# DATAFRAMES
# ------------------------------------------------------------------------------
class NAIS_Dataframe(object):

    def __init__(self, csv_file, lonMin, lonMax, latMin, latMax, stepSize):
        '''Process nais dataframe.'''
        self.headers = [
            'MMSI',
            'BaseDateTime',
            'LAT',
            'LON',
            'SOG',
            'COG',
            'Heading',
            'VesselName',
            'VesselType',
            'Status',
            'Length',
            'Width',
            'Draft',
            'Cargo'
        ]
        self.headers_required = [
            'MMSI',
            'BaseDateTime',
            'LAT',
            'LON',
            'SOG',
            'COG',
            'Heading',
            'VesselType'
        ]
        self.csv = csv_file
        self.df = pd.read_csv(self.csv, usecols=self.headers)
        self.df['BaseDateTime'] = pd.to_datetime(self.df['BaseDateTime'])

        # spatial parameters
        self.lonMin = lonMin
        self.lonMax = lonMax
        self.latMin = latMin
        self.latMax = latMax
        self.stepSize = stepSize
        self.lonRange = np.arange(
            self.lonMin,
            self.lonMax + self.stepSize,
            self.stepSize
        )
        self.latRange = np.arange(
            self.latMin,
            self.latMax + self.stepSize,
            self.stepSize
        )

    def drop_bad_data(self):
        '''Remove bad data.'''
        # Remove invalid MMSI numbers
        self.df['MMSI_Count'] = self.df['MMSI'].apply(lambda x: len(str(x)))
        self.df = self.df[self.df['MMSI_Count'] == 9].copy()
        self.df.drop(columns=['MMSI_Count'], inplace=True)

        # Remove rows with invalid heading
        self.df['Heading'] = np.where(
            self.df['Heading'] > 360,
            np.nan,
            self.df['Heading']
        )

        # Handle NOT NULL columns
        for col in self.headers_required:
            self.df[col].replace("", np.nan, inplace=True)
        self.df.dropna(how='any', subset=self.headers_required, inplace=True)

    def drop_moored(self):
        '''Drop rows with status of moored.'''
        self.df = self.df[self.df['Status'] != 'moored'].copy()
        self.df = self.df[self.df['SOG'] > 2].copy()

    def drop_spatial(self):
        '''Limit to bounding box.'''
        self.df = self.df[self.df['LON'].between(
            self.lonMin,self.lonMax, inclusive=False
        )].copy()
        self.df = self.df[self.df['LAT'].between(
            self.latMin, self.latMax, inclusive=False
        )].copy()

    def grid(self, array, i, column):
        '''Construct longitude grid.'''
        index_grid = str(i).zfill(2)
        min = round(array[i],2)
        max = round(array[i+1],2)
        cond = self.df[column].between(min, max)
        return index_grid, min, max, cond

    def add_sector(self):
        '''Add spatial sector ID.'''
        self.df.sort_values(by=['LON', 'LAT'], inplace=True)
        self.df['SectorID'] = np.nan

        # Make grid of lat and lon
        for x in range(len(self.lonRange)-1):
            ilon, min_lon, max_lon, cond_lon = self.grid(self.lonRange, x, 'LON')
            for y in range(len(self.latRange)-1):
                ilat, min_lat, max_lat, cond_lat = self.grid(self.latRange, y, 'LAT')

                # Set SectorID
                index  = "{0}.{1}".format(ilon, ilat)
                self.df['SectorID'] = np.where(
                    (cond_lon) & (cond_lat),
                    index,
                    self.df['SectorID']
                )

    def add_track(self):
        '''Add track ID for each MMSI.'''
        self.df.sort_values(by=[
            'MMSI',
            'SectorID',
            'BaseDateTime'
        ], inplace=True)

        # If time difference is greater than 6 minutes consider it new track
        timeMax = 3
        temp = self.df.groupby(['MMSI', 'SectorID'])
        self.df['Time_Delta'] = temp['BaseDateTime'].diff()
        self.df['Time_Delta'].fillna(pd.Timedelta(seconds=0), inplace=True)

        self.df['Break'] = np.where(
            self.df['Time_Delta'] > datetime.timedelta(minutes=timeMax),
            1,
            0
        )

        # The first record for each mmsi, sectorID has to be break
        self.df.loc[temp['SectorID'].head(1).index, 'Break'] = 1

        # Remove 1 row tracks
        self.df['Single_Row'] = np.where(
            (self.df['Break'] == 1) & (self.df['Break'].shift(-1) == 1),
            1,
            0
        )
        self.df = self.df[self.df['Single_Row'] == 0].copy()

        # Add incrementing track ID for each MMSI
        self.df['TrackID'] = self.df.groupby('MMSI')['Break'].cumsum()

        # Clean up
        self.df.drop(columns=['Break', 'Single_Row'], inplace=True)

    def add_ROT(self):
        '''Before any processing, add ROT field.'''
        # Get change in heading for each (mmsi, trackid)
        self.df.sort_values(by=['MMSI', 'TrackID', 'BaseDateTime'], inplace=True)
        self.df['Heading_Diff'] = self.df.groupby(['MMSI', 'TrackID'])['Heading'].diff()
        self.df['Heading_Diff'].fillna(0, inplace=True)
        # only want values betwee 0, 180
        self.df['Heading_Delta'] = 180 - abs(self.df['Heading_Diff'].abs() - 180)

        # Convert time delta to seconds
        self.df['Time_Delta'] = self.df['Time_Delta'].astype('timedelta64[s]')

        # Get change in heading per 60 seconds
        self.df['ROT'] = (self.df['Heading_Delta']/self.df['Time_Delta'])*(60)
        self.df['ROT'].fillna(0, inplace=True)
        self.df['ROT'].replace(np.inf, 0, inplace=True)

        # Clean up
        self.df.drop(
            columns=['Time_Delta', 'Heading_Diff', 'Heading_Delta'],
            inplace=True
        )

    def add_interval(self):
        '''Assign 5 minute interval to each row.'''
        self.df.sort_values(by='BaseDateTime', inplace=True)
        self.df['Interval'] = self.df['BaseDateTime'].dt.floor('1min')

    def validate_types(self):
        '''Cast to correct data types.'''
        # Replace empty elements with -1
        cols_null = list(set(self.headers) - set(self.headers_required))
        for col in cols_null:
            self.df[col].fillna(-1, inplace=True)

        # Handle string columns
        self.df['Status'].replace(-1, "undefined", inplace=True)
        self.df['VesselName'].replace(-1, "undefined", inplace=True)

        # Handle int columns
        cols_int = ['VesselType', 'Cargo', 'TrackID']
        for col in cols_int:
            self.df[col] = self.df[col].astype(int)

    def reorder_output(self):
        '''Save processed df to csv file.'''
        order = [
            'MMSI',
            'SectorID',
            'TrackID',
            'BaseDateTime',
            'Interval',
            'LAT',
            'LON',
            'SOG',
            'COG',
            'Heading',
            'ROT',
            'VesselName',
            'VesselType',
            'Status',
            'Length',
            'Width',
            'Draft',
            'Cargo'
        ]
        output = self.df.reindex(order, axis="columns")
        output.to_csv(self.csv, index=False, header=False)

class Sector_Dataframe(object):

    def __init__(self, lonmin, lonmax, latmin, latmax, stepsize):
        '''Make sector dataframe.'''
        self.grid_df = pd.DataFrame(columns=['MinLon', 'MinLat', 'MaxLon', 'MaxLat'])

        # spatial parameters
        self.lonMin = lonmin
        self.lonMax = lonmax
        self.latMin = latmin
        self.latMax = latmax
        self.stepSize = stepsize
        self.lon = np.arange(self.lonMin, self.lonMax + self.stepSize, self.stepSize)
        self.lat = np.arange(self.latMin, self.latMax + self.stepSize, self.stepSize)

        # TODO: consolidate sector and nais grid generation
    def grid(self, array, i):
        '''Construct longitude grid.'''
        index_grid = str(i).zfill(2)
        min = round(array[i],2)
        max = round(array[i+1],2)
        return index_grid, min, max

    def generate_df(self):
        '''Add spatial sector ID.'''
        for x in range(len(self.lon)-1):
            ilon, min_lon, max_lon = self.grid(self.lon, x)
            for y in range(len(self.lat)-1):
                ilat, min_lat, max_lat = self.grid(self.lat, y)

                index  = "{0}.{1}".format(ilon, ilat)
                index_row = [min_lon, min_lat, max_lon, max_lat]
                self.grid_df.loc[index] = index_row
        return self.grid_df


# ------------------------------------------------------------------------------
# TABLES
# ------------------------------------------------------------------------------
class Postgres_Table(object):

    def __init__(self, conn, table):
        '''Connect to database.'''
        self.conn = conn
        self.cur = self.conn.cursor()
        self.table = table

        # Enable PostGIS
        self.cur.execute("CREATE EXTENSION IF NOT EXISTS postgis")
        self.conn.commit()
        register(self.conn)

    def set_parallel(self, cores):
        '''Enable parallel.'''
        self.cur.execute("SET max_parallel_workers={0}".format(cores))
        self.cur.execute("SET max_parallel_workers_per_gather={0}".format(cores))
        self.conn.commit()

    def set_timezone(self):
        '''Set timezone. The data is in UTC.'''
        self.cur.execute("SET timezone = 'UTC'")
        self.conn.commit()

    def drop_table(self):
        '''Drop the table if it exists.'''
        print("Dropping table: {0}".format(self.table))
        sql = "DROP TABLE IF EXISTS {0}".format(self.table)
        self.cur.execute(sql)
        self.conn.commit()

    def copy_data(self, csv_file):
        '''Copy data into table.'''
        print('Copying {0} to database...'.format(csv_file))
        with open(csv_file, 'r') as f:
            self.cur.copy_from(f, self.table, sep=',')
            self.conn.commit()

    def create_table(self, filepath=None):
        '''Create given table.'''
        if filepath:
            cmd = "shp2pgsql -s 4326 -d {0} {1} | psql -d postgres -U postgres -q".format(filepath, self.table)
            subprocess.call(cmd, shell=True)
        else:
            sql = """
                CREATE TABLE IF NOT EXISTS {0} ({1})
            """.format(self.table, self.columns)
            self.cur.execute(sql)
            self.conn.commit()

    def add_index(self, name, field):
        print('Adding index for {0}.'.format(field))
        index = """
            CREATE INDEX IF NOT EXISTS {0}
            ON {1} ({2})
        """.format(name, self.table, field)
        self.cur.execute(index)
        self.conn.commit()

    def add_column(self, name, datatype=None, geometry=False):
        '''Add column with datatype to the table.'''
        if geometry:
            sql = """
                ALTER TABLE {0}
                ADD COLUMN IF NOT EXISTS {1} geometry({2}, 4326)
            """
        else:
            sql = """
                ALTER TABLE {0}
                ADD COLUMN IF NOT EXISTS {1} {2}
            """
        self.cur.execute(sql.format(self.table, name, datatype))
        self.conn.commit()

    def add_point(self, name, lon, lat, time=None):
        '''Add point geometry to column.'''
        if time:
            sql = """
                UPDATE {0}
                SET {1} = ST_SetSRID(ST_MakePointM({2}, {3}, {4}), 4326)
            """.format(self.table, name, lon, lat, time)
        else:
            sql = """
                UPDATE {0}
                SET {1} = ST_SetSRID(ST_MakePoint({2}, {3}), 4326)
            """.format(self.table, name, lon, lat)
        self.cur.execute(sql)
        self.conn.commit()

class Shapefile_Table(Postgres_Table):

    def __init__(self, conn, table):
        '''Connect to default database.'''
        super(Shapefile_Table, self).__init__(conn, table)
        self.cur = self.conn.cursor()

    def reduce_table(self, region):
        '''Limit shore to Pacific region.'''
        sql = "DELETE FROM {0} WHERE regions != '{1}'".format(
            self.table,
            region
        )
        self.cur.execute(sql)
        self.conn.commit()

class Grid_Table(Postgres_Table):

    def __init__(self, conn, table):
        '''Connect to default database.'''
        super(Grid_Table, self).__init__(conn, table)
        self.cur = self.conn.cursor()

        self.columns = """
            SectorID char(5) PRIMARY KEY,
            MinLon float8 NOT NULL,
            MinLat float8 NOT NULL,
            MaxLon float8 NOT NULL,
            MaxLat float8 NOT NULL
        """

    def add_points(self):
        '''Add Point geometry to the database.'''
        self.add_column('minpoint', datatype='POINT', geometry=True)
        self.add_column('maxpoint', datatype='POINT', geometry=True)

        print('Adding PostGIS POINTs to table.')
        self.add_point('minpoint', 'minlon', 'minlat')
        self.add_point('maxpoint', 'maxlon', 'maxlat')

    def make_bounding_box(self):
        '''Add polygon column in order to do spatial analysis.'''
        self.add_column('boundingbox', datatype='BOX', geometry=True)

        sql = """
            UPDATE {0}
            SET {1} = ST_SetSRID(ST_ENVELOPE({2}, {3}), 4326)
        """.format(self.table, 'boundingbox', 'minpoint', 'maxpoint')

class Points_Table(Postgres_Table):

    def __init__(self, conn, table):
        '''Connect to default database.'''
        super(Points_Table, self).__init__(conn, table)
        self.cur = self.conn.cursor()

        self.columns = """
            MMSI char(9) CHECK (char_length(MMSI) = 9) NOT NULL,
            SectorID char(5) NOT NULL,
            TrackID integer NOT NULL,
            BaseDateTime timestamp NOT NULL,
            Interval timestamp NOT NULL,
            LAT float8 NOT NULL,
            LON float8 NOT NULL,
            SOG float(4) NOT NULL,
            COG float(4) NOT NULL,
            Heading float(4) NOT NULL,
            ROT float(4) NOT NULL,
            VesselName varchar(32),
            VesselType integer NOT NULL default -1,
            Status varchar(64),
            Length float(4),
            Width float(4),
            Draft float(4),
            Cargo integer default -1
        """

    def add_geometry(self):
        '''Add PostGIS Point geometry to the database and make it index.'''
        print('Adding PostGIS POINTM to table.')
        self.add_column('geom', datatype='POINTM', geometry=True)
        self.add_point('geom', 'lon', 'lat', "date_part('epoch', basedatetime)")

    def add_tss(self):
        '''Add column marking whether the point is in the TSS or not.'''
        column = 'intss'
        self.add_column(column, datatype='boolean', geometry=False)
        self.conn.commit()
        sql = """
            UPDATE {0}
            SET {1} = ST_Contains(tss.geom, {2}.geom)
            FROM tss
        """.format(self.table, column, self.table)

    def add_indexes(self):
        '''Add indices.'''
        self.add_index("idx_tracks", "mmsi, sectorid, trackid")
        self.add_index("idx_near", "sectorid, (basedatetime::DATE), interval, mmsi")

class Tracks_Table(Postgres_Table):

    def __init__(self, conn, table):
        '''Connect to default database.'''
        super(Tracks_Table, self).__init__(conn, table)
        self.cur = self.conn.cursor()

    def add_tracks(self):
        '''Add LINESTRING for each MMSI, TrackID.'''
        sql = """
            CREATE TABLE {0} AS
            SELECT
                mmsi,
                sectorid,
                trackid,
                ST_MakeLine(geom ORDER BY basedatetime) AS track
            FROM nais_points_10
            GROUP BY mmsi, sectorid, trackid
            """.format(self.table)
        self.cur.execute(sql)
        self.conn.commit()

    def reduce_table(self):
        '''Remove tracks that cross the shoreline.'''
        sql = """
            DELETE FROM {0} AS n
            USING shore AS s
            WHERE ST_Intersects(n.track, s.geom)
        """.format(self.table)
        self.cur.execute(sql)
        self.conn.commit()

class Near_Table(Postgres_Table):

    def __init__(self, conn, table, input_table):
        '''Connect to default database.'''
        super(Near_Table, self).__init__(conn, table)
        self.cur = self.conn.cursor()
        self.input = input_table

        self.columns =  [
            'own_mmsi',
            'own_sectorid',
            'own_sog',
            'own_heading',
            'own_rot',
            'own_length',
            'own_vesseltype',
            'own_tss'
            'target_sog',
            'target_tss',
            'azimuth_deg',
            'spot_distance',
            'bearing'
        ]
        self.columnString =  ', '.join(self.columns[:-1])

    def near_points(self):
        '''Make pairs of points that happen in same sector and time interval.'''
        print('Joining nais_points with itself to make near table...')
        sql = """
            CREATE TABLE {0} AS
            SELECT
                n1.mmsi AS own_mmsi,
                n1.sectorid AS own_sectorid,
                n1.trackid AS own_trackid,
                n1.interval AS own_interval,
                n1.sog AS own_sog,
                n1.cog AS own_cog,
                n1.heading AS own_heading,
                n1.rot AS own_rot,
                n1.vesseltype AS own_vesseltype,
                n1.length AS own_length,
                n1.geom AS own_geom,
                n1.intss AS own_tss,
                n2.mmsi AS target_mmsi,
                n2.sectorid AS target_sectorid,
                n2.trackid AS target_trackid,
                n2.interval AS target_interval,
                n2.sog AS target_sog,
                n2.cog AS target_cog,
                n2.heading AS target_heading,
                n2.rot AS target_rot,
                n2.vesseltype AS target_vesseltype,
                n2.geom AS target_geom,
                n2.intss AS target_tss
                ST_Distance(n1.geom::geography, n2.geom::geography) AS spot_distance,
                DEGREES(ST_Azimuth(n1.geom, n2.geom)) AS azimuth_deg
            FROM {1} n1 INNER JOIN {2} n2
            ON n1.sectorid = n2.sectorid
            WHERE n1.basedatetime::date = n2.basedatetime::date
            AND n1.interval = n2.interval
            AND n1.mmsi != n2.mmsi
        """.format(self.table, self.input, self.input)
        self.cur.execute(sql)
        self.conn.commit()

    def near_points_dataframe(self, max_distance, sector_id):
        '''Return dataframe of near points within max distance.'''
        sql = """
            SELECT {0} FROM {1}
            WHERE spot_distance <= {2}
            AND own_sectorid = '{3}'
            AND own_sog > 5
            AND target_sog > 5
        """.format(self.columnString, self.table, max_distance, sector_id)
        self.cur.execute(sql)
        column_names = [desc[0] for desc in self.cur.description]
        return pd.DataFrame(self.cur.fetchall(), columns = column_names)

    def near_table(self, max_distance, sector_id):
        '''Create near table in pandas using max_distance = 1nm.'''
        # 1nm = 1852 meters
        df = self.near_points_dataframe(max_distance, sector_id)
        df['bearing'] = (df['azimuth_deg'] - df['own_heading']) % 360
        return df[self.columns].copy()

    def near_plot(self, max_distance, sector_id, display_points):
        '''Plot the near points all in reference to own ship.'''
        self.df_near = self.near_table(max_distance, sector_id).head(display_points)
        theta = self.df_near['bearing']
        r = self.df_near['spot_distance']
        tss = self.df_near['own_tss']
        colors = tss
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='polar')
        c = ax.scatter(theta, r, c=colors, cmap='hsv', alpha=0.75)
        plt.show()

    # def dbscan_near(self):
    # do dbscan on the POINTs
    # fit a circle to it - use as ship domain
    # use cpa distance equal to this radius

    def high_rot(self, rot_max):
        '''Select high ROT points.'''
        sql = """
            CREATE TABLE high_rot_{0} AS
            SELECT *
            FROM {1}
            WHERE rot > {2}
        """.format(str(rot_max), self.table, rot_max)
        self.cur.execute(sql)
        self.conn.commit()

class CPA_Table(Postgres_Table):

    def __init__(self, conn, table, input_table):
        '''Connect to default database.'''
        super(CPA_Table, self).__init__(conn, table)
        self.cur = self.conn.cursor()
        self.input = input_table

    def near_tracks(self):
        '''Make pairs of points that happen in same sector and time interval.'''
        print('Joining nais_tracks with itself to make near table...')
        sql = """
            CREATE TABLE {0} AS
            SELECT
                n1.mmsi AS own_mmsi,
                n1.sectorid AS own_sectorid,
                n1.trackid AS own_trackid,
                n1.track AS own_track,
                n2.mmsi AS target_mmsi,
                n2.sectorid AS target_sectorid,
                n2.trackid AS target_trackid,
                n2.track AS target_track,
                ST_ClosestPointOfApproach(n1.track, n2.track) AS cpa_point,
                ST_DistanceCPA(n1.track, n2.track) AS cpa_distance,
                ST_Angle(n1.track, n2.track) AS angle
            FROM {1} n1 INNER JOIN {2} n2
            ON n1.sectorid = n2.sectorid
            WHERE n1.mmsi != n2.mmsi
            AND ST_CPAWithin(n1.track, n2.track, 926)
        """.format(self.table, self.input, self.input)
        self.cur.execute(sql)
        self.conn.commit()
