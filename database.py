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
import multiprocessing
import numpy as np
import os
from os.path import exists, join
import osgeo.ogr
import pandas as pd
from postgis.psycopg import register
import psycopg2
from psycopg2 import sql
from retrying import retry
import shutil
import subprocess
import tempfile
import time
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
        self.csv = None

        # spatial parameters
        self.lonMin = -126.
        self.lonMax = -122.
        self.latMin = 45.
        self.latMax = 49.
        self.stepSize = 0.1

        # dataframe parameters
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

        # database parameters
        self.password = password
        self.conn = psycopg2.connect(
            host='localhost',
            dbname='postgres',
            user='postgres',
            password=self.password)


    # --------------------------------------------------------------------------
    # MAIN METHODS
    # --------------------------------------------------------------------------
    def build_tables(self):
        '''Build database of raw data.'''
        start = time.time()
        self.download_raw()

        try:
            # Build shoreline table
            print('Constructing shoreline table...')
            self.table_shore = Shapefile_Table(self.conn, 'shore')
            self.table_shore.drop_table()
            self.table_shore.create_table(filepath=self.shoreline_shp)
            self.table_shore.reduce_table()

            # Build shoreline table
            print('Constructing tss table...')
            self.table_tss = Shapefile_Table(self.conn, 'tss')
            self.table_tss.drop_table()
            self.table_tss.create_table(filepath=self.tss_shp)

            # Build nais points table
            print('Constructing nais_points table...')
            self.table_points = Points_Table(self.conn, 'nais_points')
            self.table_points.drop_table()
            self.table_points.create_table()
            self.table_points.set_timezone()
            self.table_points.set_parallel(10)

            # Copy data from temp directory and clean up
            for csv_file in glob(self.root + '\\*.csv'):
                self.csv = csv_file
                self.preprocess_csv()
                self.table_points.copy_data(self.csv)

            # Add PostGIS geometry and crete index for space and time
            self.table_points.add_geometry()
            self.table_points.add_index()

            # Create grid table
            print('Constructing grid table...')
            self.table_grid = Grid_Table(self.conn, 'grid')
            self.table_grid.drop_table()
            self.table_grid.copy_data(self.grid_csv)
            self.table_grid.add_points()
            self.table_grid.add_box()

            # Create tracks table from points table
            print('Constructing nais_tracks table...')
            self.table_tracks = Tracks_Table(self.conn, 'nais_tracks')
            self.table_tracks.drop_table()
            self.table_tracks.add_tracks()
            print('Removing tracks that cross shore...')
            self.table_tracks.reduce_table()

        except Exception as err:
            print(err)
            self.conn.rollback()
        finally:
            shutil.rmtree(self.root)
            end = time.time()
            print('Elapsed Time: {0}'.format(end-start))

    def download_raw(self):
        '''Dowload each month's raw data to a temp directory.'''
        raw = NAIS_Download(self.root, self.zone, self.year)
        for month in [str(i).zfill(2) for i in range(1, 13)]:
            raw.download_nais(month)

        shore = Shoreline_Download(self.root)
        self.shoreline_shp = shore.download_shoreline()

        tss = TSS_Download(self.root)
        self.tss_shp = tss.download_tss()

    def preprocess_csv(self):
        '''Clean real columns and write to temp dir.'''
        print('Preprocessing file {0}...'.format(self.csv))
        self.df = pd.read_csv(self.csv, usecols = self.headers)
        self.df['BaseDateTime'] = pd.to_datetime(self.df['BaseDateTime'])

        # Reduce data
        self.drop_bad_data()
        self.drop_spatial()
        self.drop_moored()

        # Add derived fields
        self.add_sector()
        self.add_track()
        self.add_ROT()

        # Handle data types
        self.validate_types()

        # Write
        self.reorder_output()


    # --------------------------------------------------------------------------
    # UTILITY METHODS
    # --------------------------------------------------------------------------
    def drop_bad_data(self):
        '''Remove bad data.'''
        # Remove invalid MMSI numbers
        self.df['MMSI_Count'] = self.df['MMSI'].apply(lambda x: len(str(x)))
        self.df = self.df[self.df['MMSI_Count'] == 9].copy()
        self.df.drop(columns=['MMSI_Count'], inplace=True)

        # Remove rows with undefined heading
        self.df['Heading'].replace(511, np.nan, inplace=True)

        # Handle NOT NULL columns
        for col in self.headers_required:
            self.df[col].replace("", np.nan, inplace=True)
        self.df.dropna(how='any', subset=self.headers_required, inplace=True)

    def drop_spatial(self):
        '''Limit to bounding box.'''
        self.df = self.df[self.df['LON'].between(
            self.lonMin,self.lonMax, inclusive=False
        )].copy()
        self.df = self.df[self.df['LAT'].between(
            self.latMin, self.latMax, inclusive=False
        )].copy()

    def drop_moored(self):
        '''Drop rows with status of moored.'''
        self.df = self.df[self.df['Status'] != 'moored'].copy()
        self.df = self.df[self.df['SOG'] > 2].copy()

    def grid_lon(self, lonarray, x):
        '''Construct longitude grid.'''
        index_lon = str(x).zfill(2)
        min_lon = round(lonarray[x],2)
        max_lon = round(lonarray[x+1],2)
        cond_lon = self.df['LON'].between(min_lon, max_lon)
        return index_lon, min_lon, max_lon, cond_lon

    def grid_lat(self, latarray, y):
        '''Construct longitude grid.'''
        index_lat = str(y).zfill(2)
        min_lat = round(latarray[y],2)
        max_lat = round(latarray[y+1],2)
        cond_lat = self.df['LAT'].between(min_lat, max_lat)
        return index_lat, min_lat, max_lat, cond_lat

    def add_sector(self):
        '''Add spatial sector ID.'''
        self.df.sort_values(by=['LON', 'LAT'], inplace=True)
        self.df['SectorID'] = np.nan

        # Make grid of lat and lon
        self.grid_df = pd.DataFrame(columns=['MinLon', 'MinLat', 'MaxLon', 'MaxLat'])
        lon = np.arange(self.lonMin, self.lonMax + self.stepSize, self.stepSize)
        lat = np.arange(self.latMin, self.latMax + self.stepSize, self.stepSize)

        for x in range(len(lon)-1):
            index_lon, min_lon, max_lon, cond_lon = self.grid_lon(lon, x)
            for y in range(len(lat)-1):
                index_lat, min_lat, max_lat, cond_lat = self.grid_lat(lat, y)

                # Set SectorID
                index  = "{0}.{1}".format(index_lon, index_lat)
                self.df['SectorID'] = np.where(
                    (cond_lon) & (cond_lat),
                    index,
                    self.df['SectorID']
                )

                # Create grid dataframe
                index_row = [min_lon, min_lat, max_lon, max_lat]
                self.grid_df.loc[index] = index_row

        self.grid_csv = join(self.root, 'grid_table.csv')
        self.grid_df.drop_duplicates(inplace=True)
        self.grid_df.to_csv(self.grid_csv, index=False, header=False)

    def add_track(self):
        '''Add track ID for each MMSI.'''
        self.df.sort_values(by=[
            'MMSI',
            'SectorID',
            'BaseDateTime'
        ], inplace=True)

        # If time difference is greater than 6 minutes consider it new track
        temp = self.df.groupby(['MMSI', 'SectorID'])
        self.df['Time_Delta'] = temp['BaseDateTime'].diff()
        self.df['Time_Delta'].fillna(pd.Timedelta(seconds=0), inplace=True)

        self.df['Break'] = np.where(
            self.df['Time_Delta'] > datetime.timedelta(minutes=6),
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
        self.df['Heading_Delta'] = 180 - abs(self.df['Heading_Diff'].abs() - 180)

        # Convert time delta to seconds
        self.df['Time_Delta'] = self.df['Time_Delta'].astype('timedelta64[s]')

        # Get change in heading per 60 seconds
        self.df['ROT'] = (self.df['Heading_Delta']/self.df['Time_Delta'])*(60)
        self.df['ROT'].fillna(0, inplace=True)

        # Clean up
        self.df.drop(
            columns=['Time_Delta', 'Heading_Diff', 'Heading_Delta'],
            inplace=True
        )

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
            print('The Shoreline download already exists')
            return output

        print('Downloading the US Shoreline shapefile.')
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
            print('The TSS download already exists')
            return output

        print('Downloading the TSS shapefile.')
        download = requests.get(self.url)
        zfile = zipfile.ZipFile(io.BytesIO(download.content))
        # zfile.extractall(self.root)

        # zfile = download_url(self.url, self.root, '.zip')
        extract_zip(zfile, self.root)
        os.remove(zfile)
        return output

class NAIS_Download(object):

    '''
    Download raw NAIS data from MarineCadastre to local temp directory.
    '''

    def __init__(self, root, zone, year):
        self.root = root
        self.year = year
        self.zone = zone

    @retry
    def download_nais(self, month):
        '''Download zip file and extract to temp directory.'''
        self.month = month
        self.param_file = (self.year, self.month, self.zone)
        self.csv_raw = 'AIS_%s_%s_Zone%s.csv' % self.param_file

        # Download URL
        param = (self.year, self.year, self.month, self.zone)
        self.url = 'https://coast.noaa.gov/htdata/CMSP/AISDataHandler/%s/AIS_%s_%s_Zone%s.zip' % param

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
            cmd = "shp2pgsql -s 4326 {0} {1} | psql -d postgres -U postgres -q".format(filepath, self.table)
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
            ON {1} USING GIST({2})
        """.format(name, self.table, field)
        self.cur.execute(index)
        self.conn.commit()

class Shapefile_Table(Postgres_Table):

    def __init__(self, conn, table):
        '''Connect to default database.'''
        super(Shapefile_Table, self).__init__(conn, table)
        self.cur = self.conn.cursor()

    def reduce_table(self):
        # Limit shore to Pacific region
        sql = "DELETE FROM {0} WHERE regions != 'P'".format(self.table)
        self.cur.execute(sql)
        self.conn.commit()

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
        add_column = """
            ALTER TABLE {0}
            ADD COLUMN geom geometry(POINTM, 4326)
        """.format(self.table)
        self.cur.execute(add_column)

        print('Adding PostGIS POINTM to table.')
        add_geom = """
            UPDATE {0}
            SET geom = ST_SetSRID(
                ST_MakePointM(lon, lat, date_part('epoch', basedatetime)),
                4326
            )
        """.format(self.table)
        self.cur.execute(add_geom)
        self.conn.commit()

    def add_index(self):
        '''Add spatial, time, and MMSI indices.'''
        self.add_index('idx_point', 'GIST(geom)')
        self.add_index('idx_time', 'basedatetime')

        print('Adding MMSI index.')
        index_mmsi =  """
            CREATE INDEX IF NOT EXISTS idx_mmsi
            ON {0} (mmsi)
        """.format(self.table)
        self.cur.execute(index_mmsi)
        self.conn.commit()

class Tracks_Table(Postgres_Table):

    def __init__(self, conn, table):
        '''Connect to default database.'''
        super(Tracks_Table, self).__init__(conn, table)
        self.cur = self.conn.cursor()

    def buffer_shore(self):
        '''Keep only nais points not within 1 nautical mile of the shore.'''
        sql = """
            CREATE materialized view IF NOT EXISTS buffer AS
            SELECT *
            FROM (
                SELECT
                    n.mmsi,
                    n.sectorid,
                    n.trackid,
                    n.basedatetime,
                    n.lon,
                    n.lat,
                    n.geom point,
                    s.geom line
                FROM nais_points n, shor s
                WHERE not ST_DWithin(n.geom, s.geom, 1852)
            ) as shoreBuffer
        """
        self.cur.execute(sql)
        self.conn(commit)

    def add_tracks(self):
        '''Add LINESTRING for each MMSI, TrackID.'''
        sql = """
            CREATE TABLE nais_tracks AS
            SELECT
                mmsi,
                sectorid,
                trackid,
                ST_MakeLine(geom ORDER BY basedatetime) AS track
            FROM nais_points
            GROUP BY mmsi, sectorid, trackid
            """
        self.cur.execute(sql)
        self.conn.commit()

    def reduce_table(self):
        # Limit shore to Pacific region
        sql = """
            DELETE FROM nais_tracks n, shore s
            WHERE ST_Intersects(n.track, s.geom)
        """
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
        '''Add PostGIS Point geometry to the database.'''
        sql = """
            ALTER TABLE {0}
            ADD COLUMN {1} geometry(POINTM, 4326)
        """
        self.cur.execute(sql.format(self.table, 'minpoint'))
        self.cur.execute(sql.format(self.table, 'maxpoint'))
        self.conn.commit()

        print('Adding PostGIS POINTM to table.')
        sql = """
            UPDATE {0}
            SET {1} = ST_SetSRID(
                ST_MakePoint({2}, {3}),
                4326
            )
        """
        self.cur.execute(sql.format(self.table, 'minpoint', 'minlon', 'minlat'))
        self.cur.execute(sql.format(self.table, 'maxpoint', 'maxlon', 'maxlat'))
        self.conn.commit()

    def add_box(self):
        '''Add PostGIS Box2D geometry to the database.'''
        sql = """
            ALTER TABLE {0}
            ADD COLUMN {1} geometry(Box2D, 4326)
        """
        self.cur.execute(sql.format(self.table, 'box'))
        self.conn.commit()

        sql = """
            UPDATE {0}
            SET {1} = ST_SetSRID(
                ST_Envelope({0},{1}),
                4326
            )
        """
        self.cur.execute(sql.format(self.table, 'box', 'minpoint', 'maxpoint'))
        self.conn.commit()
