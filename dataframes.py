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
from glob import glob
import io
import matplotlib.pyplot as plt
from multiprocessing.dummy import Pool
import numpy as np
import os
from os.path import dirname, exists, join#!/usr/bin/env python
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
from glob import glob
import io
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
import zipfile
import yaml

from . import time_all
from . import download_url, extract_zip, find_file
from . import dataframes
from .downloads import TSS_Download, Shoreline_Download, NAIS_Download


# ------------------------------------------------------------------------------
# MAIN CLASS
# ------------------------------------------------------------------------------
@time_all
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

        self.zone = self.parameters['zone']
        self.lonMin = self.parameters['lonMin']
        self.lonMax = self.parameters['lonMax']
        self.latMin = self.parameters['latMin']
        self.latMax = self.parameters['latMax']
        self.stepSize = self.parameters['stepSize']

        # time parameters
        # self.months = [str(i).zfill(2) for i in range(1, 13)]
        self.months = ['01']

        # database parameters
        self.conn = psycopg2.connect(
            host='localhost',
            dbname='postgres',
            user='postgres',
            password=self.password)

        # tables
        self.table_shore = Shapefile_Table(
            self.conn,
            'shore_{0}'.format(self.zone)
        )
        self.table_tss = Shapefile_Table(
            self.conn,
            'tss_{0}'.format(self.zone)
        )
        self.table_points = Points_Table(
            self.conn,
            'nais_points_{0}'.format(self.zone)
        )
        self.table_rot = ROT_Table(
            self.conn,
            'nais_rot_{0}'.format(self.zone)
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
        self.table_cpa = Interactions_Table(
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

    def download_raw(self):
        '''Dowload raw data to a temp directory.'''
        raw = NAIS_Download(self.root, self.city, self.year)
        for month in self.months:
            raw.download_nais(month)
        raw.clean_up()


    # BUILD DATABASE -----------------------------------------------------------
    def build_tables(self):
        '''Build database of raw data.'''
        start = time.time()
        try:
            # Environmental
            self.build_shore()
            self.build_tss()
            self.build_grid()

            # Points
            self.build_nais_points()
            self.table_rot.drop_table()
            self.table_rot.select_rot(100)

            # Tracks
            self.build_nais_tracks()
            self.build_nais_interactions()

            # Make near points table
            self.table_near.drop_table()
            self.table_near.near_points()


        except Exception as err:
            print(err)
            self.conn.rollback()
            self.conn.close()
        finally:
            # shutil.rmtree(self.root)
            end = time.time()
            print('Elapsed Time: {0} minutes'.format((end-start)/60))

    def build_shore(self):
        '''Construct shoreline table.'''
        print('Constructing shoreline table...')
        shore = Shoreline_Download(self.root)
        self.shoreline_shp = shore.download_shoreline()
        self.table_shore.create_table(filepath=self.shoreline_shp)
        self.table_shore.reduce_table('regions', self.parameters['region'])

    def build_tss(self):
        '''Construct TSS table.'''
        print('Constructing tss table...')
        tss = TSS_Download(self.root)
        self.tss_shp = tss.download_tss()
        self.table_tss.create_table(filepath=self.tss_shp)
        self.table_tss.reduce_table('objl', self.parameters['tss'])

    def build_grid(self):
        '''Create grid table.'''
        print('Constructing grid table...')
        self.grid_df = dataframes.Sector_Dataframe(
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
        self.table_grid.make_bounding_box()

    def build_nais_points(self):
        '''Build nais points table.'''
        print('Constructing nais_points table...')
        self.download_raw()

        raw = NAIS_Download(self.root, self.city, self.year)
        with Pool(processes=12) as pool:
            pool.map(raw.preprocess_nais, self.months)

        self.table_points.drop_table()
        self.table_points.create_table()
        self.table_points.set_parallel(20)
        self.table_points.set_timezone()

        for csv_file in self.nais_csvs:
            self.table_points.copy_data(csv_file)

        self.table_points.add_geometry()
        self.table_points.add_indexes()
        self.table_points.add_tss(self.table_tss.table)

    def build_nais_tracks(self):
        '''Create tracks table from points table.'''
        print('Constructing nais_tracks table...')
        self.table_tracks.drop_table()
        self.table_tracks.convert_to_tracks(self.table_points.table)

        print('Removing tracks that cross shore...')
        self.table_tracks.reduce_table(self.table_shore.table)

    def build_nais_interactions(self):
        '''Create table to generate pair-wise cpa.'''
        self.table_cpa.drop_table()
        self.table_cpa.interaction_tracks()
        self.table_cpa.interaction_cpa()


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
        '''Enable parallel - not sure if necessary.'''
        self.cur.execute("SET dynamic_shared_memory_type='windows'")
        self.cur.execute("SET max_worker_processes={0}".format(cores))
        self.cur.execute("SET max_parallel_workers={0}".format(cores))
        self.cur.execute("SET max_parallel_workers_per_gather={0}".format(cores/2))
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

    def add_column(self, name, datatype=None, geometry=False, default=None):
        '''Add column with datatype to the table.'''
        sql_alter = """
            ALTER TABLE {0} ADD COLUMN IF NOT EXISTS {1}
            """.format(self.table, name)

        # Handle geometry types
        if geometry:
            sql_type = """geometry({0}, 4326)""".format(datatype)
        else:
            sql_type = """{0} """.format(datatype)

        sql = sql_alter + sql_type
        # Handle default data types
        if default:
            sql_default = """DEFAULT {0}""".format(default)
            sql = sql + sql_default

        self.cur.execute(sql)
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

    def reduce_table(self, column, value):
        '''Drop rows on one NOT EQUAL TO condition.'''
        print('Dropping {0} != {1} from {2}'.format(column, value, self.table))
        if isinstance(value, str):
            sql_delete = "DELETE FROM {0} WHERE {1} != '{2}'"
        else:
            sql_delete = "DELETE FROM {0} WHERE {1} != {2}"
        sql = sql_delete.format(self.table, column, value)
        self.cur.execute(sql)
        self.conn.commit()

    def table_dataframe(self, select_col=None, where_cond=None):
        '''Return dataframe.'''
        if select_col is None:
            select_col = '*'
        if where_cond is None:
            sql = """
                SELECT {0}
                FROM {1}
            """.format(select_col, self.table)
        else:
            sql = """
                SELECT {0}
                FROM {1}
                WHERE {2}
            """.format(select_col, self.table, where_cond)

        self.cur.execute(sql)
        column_names = [desc[0] for desc in self.cur.description]
        return pd.DataFrame(self.cur.fetchall(), columns=column_names)

class Shapefile_Table(Postgres_Table):

    def __init__(self, conn, table):
        '''Connect to default database.'''
        super(Shapefile_Table, self).__init__(conn, table)
        self.cur = self.conn.cursor()

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
        self.add_column('leftBottom', datatype='POINT', geometry=True)
        self.add_column('leftTop', datatype='POINT', geometry=True)
        self.add_column('rightTop', datatype='POINT', geometry=True)
        self.add_column('rightBottom', datatype='POINT', geometry=True)

        print('Adding PostGIS POINTs to {0}...'.format(self.table))
        self.add_point('leftBottom', 'minlon', 'minlat')
        self.add_point('leftTop', 'minlon', 'maxlat')
        self.add_point('rightTop', 'maxlon', 'maxlat')
        self.add_point('rightBottom', 'maxlon', 'minlat')

    def make_bounding_box(self):
        '''Add polygon column in order to do spatial analysis.'''
        print('Adding PostGIS POLYGON to {0}...'.format(self.table))
        self.add_column('boundingbox', datatype='Polygon', geometry=True)
        sql = """
            UPDATE {0}
            SET {1} = ST_SetSRID(ST_MakePolygon(
                ST_MakeLine(array[{2}, {3}, {4}, {5}, {6}])
            ), 4326)
        """.format(
            self.table,
            'boundingbox',
            'leftBottom',
            'leftTop',
            'rightTop',
            'rightBottom',
            'leftBottom'
            )
        self.cur.execute(sql)
        self.conn.commit()

class Points_Table(Postgres_Table):

    def __init__(self, conn, table):
        '''Connect to default database.'''
        super(Points_Table, self).__init__(conn, table)
        self.cur = self.conn.cursor()

        self.columns = """
            MMSI char(9) CHECK (char_length(MMSI) = 9) NOT NULL,
            BaseDateTime timestamp NOT NULL,
            Track_MMSI integer NOT NULL,
            LAT float8 NOT NULL,
            LON float8 NOT NULL,
            SOG float(4) NOT NULL,
            COG float(4) NOT NULL,
            Heading float(4) NOT NULL,
            Step_ROT float(4) NOT NULL,
            Step_Acceleration float(4),
            Stop boolean,
            step_Displacement flaot(4),
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

    def add_indexes(self):
        '''Add indices.'''
        self.add_index("idx_geom", "geom")
        # self.add_index("idx_tracks", "mmsi, sectorid, trackid")
        # self.add_index("idx_near", "sectorid, basedatetime, mmsi")

    def add_tss(self, tss):
        '''Add column marking whether the point is in the TSS or not.'''
        name = 'in_tss'
        print('Adding {0} to {1}'.format(name, self.table))
        self.add_column(name, datatype='boolean', default='FALSE')

        sql = """
            UPDATE {0}
            SET {1} = TRUE
            FROM (
                SELECT points.lat, points.lon
                FROM {2} AS points
                RIGHT JOIN {3} AS polygons
                ON ST_Contains(polygons.geom, points.geom)
            ) as tssTest
            WHERE {4}.lat=tssTest.lat
            AND {5}.lon=tssTest.lon
        """.format(self.table, name, self.table, tss, self.table, self.table)
        self.cur.execute(sql)
        self.conn.commit()

class Tracks_Table(Postgres_Table):

    def __init__(self, conn, table):
        '''Connect to default database.'''
        super(Tracks_Table, self).__init__(conn, table)
        self.cur = self.conn.cursor()

    def convert_to_tracks(self, points):
        '''Add LINESTRING for each MMSI, TrackID.'''
        sql = """
            CREATE TABLE {0} AS
            SELECT
                mmsi,
                length,
                width,
                vesseltype,
                trackid,
                min(basedatetime) AS start_time,
                max(basedatetime) AS end_time,
                max(basedatetime) - min(basedatetime) AS duration,
                ST_MakeLine(geom ORDER BY basedatetime) AS track
            FROM {1}
            GROUP BY mmsi, length, width, vesseltype, trackid
            """.format(self.table, points)
        self.cur.execute(sql)
        self.conn.commit()

    def reduce_table(self, shore_table):
        '''Remove tracks that cross the shoreline.'''
        # sql = """
        #     DELETE FROM {0} AS n
        #     USING {1} AS s
        #     WHERE ST_Intersects(n.track, s.geom)
        # """.format(self.table, shore_table)
        sql = """
            DELETE FROM {0}
            WHERE duration < '00:15:00'
        """.format(self.table)
        self.cur.execute(sql)
        self.conn.commit()

class Interactions_Table(Postgres_Table):

    def __init__(self, conn, table, input_table):
        '''Connect to default database.'''
        super(Interactions_Table, self).__init__(conn, table)
        self.cur = self.conn.cursor()
        self.input = input_table

    def interaction_tracks(self):
        '''Make pairs of tracks that happen in same time interval.'''
        print('Joining nais_tracks with itself to make near table...')
        sql = """
            CREATE TABLE {0} AS
            SELECT
                n1.mmsi AS own_mmsi,
                n1.trackid AS own_trackid,
                n1.track AS own_track,
                n2.mmsi AS target_mmsi,
                n2.trackid AS target_trackid,
                n2.track AS target_track,
                ST_ClosestPointOfApproach(n1.track, n2.track) AS cpa_point,
                ST_DistanceCPA(n1.track::geometry, n2.track::geometry) AS cpa_distance
            FROM {1} n1 LEFT JOIN {2} n2
            ON n1.start_time between n2.start_time and n2.end_time
            AND ST_DWithin(n1.trackid::geometry, n2.trackid::geometry, 18520)
            AND n1.mmsi != n2.mmsi
        """.format(self.table, self.input, self.input)
        self.cur.execute(sql)
        self.conn.commit()

        sql_delete = """
            DELETE FROM {0} WHERE cpa_point IS NULL
            """.format(self.table)
        self.cur.execute(sql_delete)
        self.conn.commit()


    def interaction_cpa(self):
        '''Add CPA point, track points, cpa_distance, and time.'''
        self.add_column('cpa_pointm', 'POINTM', geometry=True)
        sql_pointm = """
            UPDATE {0}
            SET {1} = ST_Force3DM(cpa_point)
        """.format(self.table, 'cpa_pointm')

        self.add_column('own_point', 'POINTM', geometry=True)
        self.add_column('target_point', 'POINTM', geometry=True)
        sql_point = """
            UPDATE {0}
            SET {1} = ST_Force3DM(
                ST_GeometryN(
                    ST_LocateAlong({2}, cpa_point),
                1)
            )
        """
        self.cur.execute(sql_point.format(self.table, 'own_point', 'own_track'))
        self.cur.execute(sql_point.format(self.table, 'target_point', 'target_track'))

        self.add_column('cpa_time', 'TIMESTAMP', geometry=False)
        sql_time = """
            UPDATE {0}
            SET {1} = to_timestamp(cpa_point)
        """.format(self.table, 'cpa_time')
        self.cur.execute(sql_time)

        self.add_column('point_distance', 'FLOAT(4)', geometry=False)
        sql_distance = """
            UPDATE {0}
            SET {1} = ST_Distance(own_point::geography, target_point::geography)
        """.format(self.table, 'point_distance')
        self.cur.execute(sql_distance)
        self.conn.commit()


# -----------------------------------------------------------------------------
class ROT_Table(Postgres_Table):

    def __init__(self, conn, table):
        '''Connect to default database.'''
        super(ROT_Table, self).__init__(conn, table)
        self.cur = self.conn.cursor()

    def select_rot(self, min_rot):
        self.min_rot = min_rot
        sql = """
            CREATE TABLE {0} AS
            SELECT *
            FROM nais_points_10
            WHERE rot > {1}
        """.format(self.table, self.min_rot)
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
            'own_tss',
            'target_mmsi',
            'target_sectorid',
            'target_sog',
            'target_heading',
            'target_rot',
            'target_vesseltype',
            'target_tss',
            'azimuth_deg',
            'point_distance',
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
                n1.sog AS own_sog,
                n1.cog AS own_cog,
                n1.heading AS own_heading,
                n1.rot AS own_rot,
                n1.vesseltype AS own_vesseltype,
                n1.length AS own_length,
                n1.geom AS own_geom,
                n1.in_tss AS own_tss,
                n2.mmsi AS target_mmsi,
                n2.sectorid AS target_sectorid,
                n2.trackid AS target_trackid,
                n2.sog AS target_sog,
                n2.cog AS target_cog,
                n2.heading AS target_heading,
                n2.rot AS target_rot,
                n2.vesseltype AS target_vesseltype,
                n2.length AS target_length,
                n2.geom AS target_geom,
                n2.in_tss AS target_tss,
                ST_Distance(ST_Transform(n1.geom, 7260), ST_Transform(n2.geom, 7260)) AS point_distance,
                DEGREES(ST_Azimuth(n1.geom, n2.geom)) AS azimuth_deg
            FROM {1} n1 INNER JOIN {2} n2
            ON n1.sectorid = n2.sectorid
            WHERE n1.basedatetime = n2.basedatetime
            AND n1.mmsi != n2.mmsi
        """.format(self.table, self.input, self.input)
        self.cur.execute(sql)
        self.conn.commit()

    def near_points_dataframe(self, max_distance):
        '''Return dataframe of near points within max distance.'''
        cond = 'point_distance <= {0}'.format(max_distance)
        return self.table_dataframe(self.columnString, cond)

    def near_table(self, max_distance):
        '''Create near table in pandas using max_distance = 1nm.'''
        # 1nm = 1852 meters
        df = self.near_points_dataframe(max_distance)
        df['bearing'] = (df['azimuth_deg'] - df['own_heading']) % 360
        return df[self.columns].copy()

    def near_plot(self, max_distance, display_points):
        '''Plot the near points all in reference to own ship.'''
        self.df_near = self.near_table(max_distance).head(display_points)
        self.df_near = self.df_near[
            (self.df_near['own_sog'] >10) & (self.df_near['target_sog'] > 10)].copy()
        theta = np.array(self.df_near['bearing'])
        r = np.array(self.df_near['point_distance'])
        tss = np.array(self.df_near['bearing'])
        # colors = theta.apply(np.radians)
        colors = tss
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='polar')
        c = ax.scatter(theta, r, c=colors, cmap='hsv', alpha=0.75, s=1)
        # plt.legend(loc='upper left')
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
import zipfile
import yaml

from . import time_all
from . import download_url, extract_zip, find_file
from . import dataframes
from .downloads import TSS_Download, Shoreline_Download, NAIS_Download


# ------------------------------------------------------------------------------
# MAIN CLASS
# ------------------------------------------------------------------------------
@time_all
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

        self.zone = self.parameters['zone']
        self.lonMin = self.parameters['lonMin']
        self.lonMax = self.parameters['lonMax']
        self.latMin = self.parameters['latMin']
        self.latMax = self.parameters['latMax']
        self.stepSize = self.parameters['stepSize']

        # time parameters
        # self.months = [str(i).zfill(2) for i in range(1, 13)]
        self.months = ['01']

        # database parameters
        self.conn = psycopg2.connect(
            host='localhost',
            dbname='postgres',
            user='postgres',
            password=self.password)

        # tables
        self.table_shore = Shapefile_Table(
            self.conn,
            'shore_{0}'.format(self.zone)
        )
        self.table_tss = Shapefile_Table(
            self.conn,
            'tss_{0}'.format(self.zone)
        )
        self.table_points = Points_Table(
            self.conn,
            'nais_points_{0}'.format(self.zone)
        )
        self.table_rot = ROT_Table(
            self.conn,
            'nais_rot_{0}'.format(self.zone)
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
        self.table_cpa = Interactions_Table(
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

    def download_raw(self):
        '''Dowload raw data to a temp directory.'''
        raw = NAIS_Download(self.root, self.city, self.year)
        for month in self.months:
            raw.download_nais(month)
        raw.clean_up()


    # BUILD DATABASE -----------------------------------------------------------
    def build_tables(self):
        '''Build database of raw data.'''
        start = time.time()
        try:
            # Environmental
            self.build_shore()
            self.build_tss()
            self.build_grid()

            # Points
            self.build_nais_points()
            self.table_rot.drop_table()
            self.table_rot.select_rot(100)

            # Tracks
            self.build_nais_tracks()
            self.build_nais_interactions()

            # Make near points table
            self.table_near.drop_table()
            self.table_near.near_points()


        except Exception as err:
            print(err)
            self.conn.rollback()
            self.conn.close()
        finally:
            # shutil.rmtree(self.root)
            end = time.time()
            print('Elapsed Time: {0} minutes'.format((end-start)/60))

    def build_shore(self):
        '''Construct shoreline table.'''
        print('Constructing shoreline table...')
        shore = Shoreline_Download(self.root)
        self.shoreline_shp = shore.download_shoreline()
        self.table_shore.create_table(filepath=self.shoreline_shp)
        self.table_shore.reduce_table('regions', self.parameters['region'])

    def build_tss(self):
        '''Construct TSS table.'''
        print('Constructing tss table...')
        tss = TSS_Download(self.root)
        self.tss_shp = tss.download_tss()
        self.table_tss.create_table(filepath=self.tss_shp)
        self.table_tss.reduce_table('objl', self.parameters['tss'])

    def build_grid(self):
        '''Create grid table.'''
        print('Constructing grid table...')
        self.grid_df = dataframes.Sector_Dataframe(
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
        self.table_grid.make_bounding_box()

    def build_nais_points(self):
        '''Build nais points table.'''
        print('Constructing nais_points table...')
        self.download_raw()

        raw = NAIS_Download(self.root, self.city, self.year)
        with Pool(processes=12) as pool:
            pool.map(raw.preprocess_nais, self.months)

        self.table_points.drop_table()
        self.table_points.create_table()
        self.table_points.set_parallel(20)
        self.table_points.set_timezone()

        for csv_file in self.nais_csvs:
            self.table_points.copy_data(csv_file)

        self.table_points.add_geometry()
        self.table_points.add_indexes()
        self.table_points.add_tss(self.table_tss.table)

    def build_nais_tracks(self):
        '''Create tracks table from points table.'''
        print('Constructing nais_tracks table...')
        self.table_tracks.drop_table()
        self.table_tracks.convert_to_tracks(self.table_points.table)

        print('Removing tracks that cross shore...')
        self.table_tracks.reduce_table(self.table_shore.table)

    def build_nais_interactions(self):
        '''Create table to generate pair-wise cpa.'''
        self.table_cpa.drop_table()
        self.table_cpa.interaction_tracks()
        self.table_cpa.interaction_cpa()


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
        '''Enable parallel - not sure if necessary.'''
        self.cur.execute("SET dynamic_shared_memory_type='windows'")
        self.cur.execute("SET max_worker_processes={0}".format(cores))
        self.cur.execute("SET max_parallel_workers={0}".format(cores))
        self.cur.execute("SET max_parallel_workers_per_gather={0}".format(cores/2))
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

    def add_column(self, name, datatype=None, geometry=False, default=None):
        '''Add column with datatype to the table.'''
        sql_alter = """
            ALTER TABLE {0} ADD COLUMN IF NOT EXISTS {1}
            """.format(self.table, name)

        # Handle geometry types
        if geometry:
            sql_type = """geometry({0}, 4326)""".format(datatype)
        else:
            sql_type = """{0} """.format(datatype)

        sql = sql_alter + sql_type
        # Handle default data types
        if default:
            sql_default = """DEFAULT {0}""".format(default)
            sql = sql + sql_default

        self.cur.execute(sql)
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

    def reduce_table(self, column, value):
        '''Drop rows on one NOT EQUAL TO condition.'''
        print('Dropping {0} != {1} from {2}'.format(column, value, self.table))
        if isinstance(value, str):
            sql_delete = "DELETE FROM {0} WHERE {1} != '{2}'"
        else:
            sql_delete = "DELETE FROM {0} WHERE {1} != {2}"
        sql = sql_delete.format(self.table, column, value)
        self.cur.execute(sql)
        self.conn.commit()

    def table_dataframe(self, select_col=None, where_cond=None):
        '''Return dataframe.'''
        if select_col is None:
            select_col = '*'
        if where_cond is None:
            sql = """
                SELECT {0}
                FROM {1}
            """.format(select_col, self.table)
        else:
            sql = """
                SELECT {0}
                FROM {1}
                WHERE {2}
            """.format(select_col, self.table, where_cond)

        self.cur.execute(sql)
        column_names = [desc[0] for desc in self.cur.description]
        return pd.DataFrame(self.cur.fetchall(), columns=column_names)

class Shapefile_Table(Postgres_Table):

    def __init__(self, conn, table):
        '''Connect to default database.'''
        super(Shapefile_Table, self).__init__(conn, table)
        self.cur = self.conn.cursor()

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
        self.add_column('leftBottom', datatype='POINT', geometry=True)
        self.add_column('leftTop', datatype='POINT', geometry=True)
        self.add_column('rightTop', datatype='POINT', geometry=True)
        self.add_column('rightBottom', datatype='POINT', geometry=True)

        print('Adding PostGIS POINTs to {0}...'.format(self.table))
        self.add_point('leftBottom', 'minlon', 'minlat')
        self.add_point('leftTop', 'minlon', 'maxlat')
        self.add_point('rightTop', 'maxlon', 'maxlat')
        self.add_point('rightBottom', 'maxlon', 'minlat')

    def make_bounding_box(self):
        '''Add polygon column in order to do spatial analysis.'''
        print('Adding PostGIS POLYGON to {0}...'.format(self.table))
        self.add_column('boundingbox', datatype='Polygon', geometry=True)
        sql = """
            UPDATE {0}
            SET {1} = ST_SetSRID(ST_MakePolygon(
                ST_MakeLine(array[{2}, {3}, {4}, {5}, {6}])
            ), 4326)
        """.format(
            self.table,
            'boundingbox',
            'leftBottom',
            'leftTop',
            'rightTop',
            'rightBottom',
            'leftBottom'
            )
        self.cur.execute(sql)
        self.conn.commit()

class Points_Table(Postgres_Table):

    def __init__(self, conn, table):
        '''Connect to default database.'''
        super(Points_Table, self).__init__(conn, table)
        self.cur = self.conn.cursor()

        self.columns = """
            MMSI char(9) CHECK (char_length(MMSI) = 9) NOT NULL,
            BaseDateTime timestamp NOT NULL,
            Track_MMSI integer NOT NULL,
            LAT float8 NOT NULL,
            LON float8 NOT NULL,
            SOG float(4) NOT NULL,
            COG float(4) NOT NULL,
            Heading float(4) NOT NULL,
            Step_ROT float(4) NOT NULL,
            Step_Acceleration float(4),
            Stop boolean,
            step_Displacement flaot(4),
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

    def add_indexes(self):
        '''Add indices.'''
        self.add_index("idx_geom", "geom")
        # self.add_index("idx_tracks", "mmsi, sectorid, trackid")
        # self.add_index("idx_near", "sectorid, basedatetime, mmsi")

    def add_tss(self, tss):
        '''Add column marking whether the point is in the TSS or not.'''
        name = 'in_tss'
        print('Adding {0} to {1}'.format(name, self.table))
        self.add_column(name, datatype='boolean', default='FALSE')

        sql = """
            UPDATE {0}
            SET {1} = TRUE
            FROM (
                SELECT points.lat, points.lon
                FROM {2} AS points
                RIGHT JOIN {3} AS polygons
                ON ST_Contains(polygons.geom, points.geom)
            ) as tssTest
            WHERE {4}.lat=tssTest.lat
            AND {5}.lon=tssTest.lon
        """.format(self.table, name, self.table, tss, self.table, self.table)
        self.cur.execute(sql)
        self.conn.commit()

class Tracks_Table(Postgres_Table):

    def __init__(self, conn, table):
        '''Connect to default database.'''
        super(Tracks_Table, self).__init__(conn, table)
        self.cur = self.conn.cursor()

    def convert_to_tracks(self, points):
        '''Add LINESTRING for each MMSI, TrackID.'''
        sql = """
            CREATE TABLE {0} AS
            SELECT
                mmsi,
                length,
                width,
                vesseltype,
                trackid,
                min(basedatetime) AS start_time,
                max(basedatetime) AS end_time,
                max(basedatetime) - min(basedatetime) AS duration,
                ST_MakeLine(geom ORDER BY basedatetime) AS track
            FROM {1}
            GROUP BY mmsi, length, width, vesseltype, trackid
            """.format(self.table, points)
        self.cur.execute(sql)
        self.conn.commit()

    def reduce_table(self, shore_table):
        '''Remove tracks that cross the shoreline.'''
        # sql = """
        #     DELETE FROM {0} AS n
        #     USING {1} AS s
        #     WHERE ST_Intersects(n.track, s.geom)
        # """.format(self.table, shore_table)
        sql = """
            DELETE FROM {0}
            WHERE duration < '00:15:00'
        """.format(self.table)
        self.cur.execute(sql)
        self.conn.commit()

class Interactions_Table(Postgres_Table):

    def __init__(self, conn, table, input_table):
        '''Connect to default database.'''
        super(Interactions_Table, self).__init__(conn, table)
        self.cur = self.conn.cursor()
        self.input = input_table

    def interaction_tracks(self):
        '''Make pairs of tracks that happen in same time interval.'''
        print('Joining nais_tracks with itself to make near table...')
        sql = """
            CREATE TABLE {0} AS
            SELECT
                n1.mmsi AS own_mmsi,
                n1.trackid AS own_trackid,
                n1.track AS own_track,
                n2.mmsi AS target_mmsi,
                n2.trackid AS target_trackid,
                n2.track AS target_track,
                ST_ClosestPointOfApproach(n1.track, n2.track) AS cpa_point,
                ST_DistanceCPA(n1.track::geometry, n2.track::geometry) AS cpa_distance
            FROM {1} n1 LEFT JOIN {2} n2
            ON n1.start_time between n2.start_time and n2.end_time
            AND ST_DWithin(n1.trackid::geometry, n2.trackid::geometry, 18520)
            AND n1.mmsi != n2.mmsi
        """.format(self.table, self.input, self.input)
        self.cur.execute(sql)
        self.conn.commit()

        sql_delete = """
            DELETE FROM {0} WHERE cpa_point IS NULL
            """.format(self.table)
        self.cur.execute(sql_delete)
        self.conn.commit()


    def interaction_cpa(self):
        '''Add CPA point, track points, cpa_distance, and time.'''
        self.add_column('cpa_pointm', 'POINTM', geometry=True)
        sql_pointm = """
            UPDATE {0}
            SET {1} = ST_Force3DM(cpa_point)
        """.format(self.table, 'cpa_pointm')

        self.add_column('own_point', 'POINTM', geometry=True)
        self.add_column('target_point', 'POINTM', geometry=True)
        sql_point = """
            UPDATE {0}
            SET {1} = ST_Force3DM(
                ST_GeometryN(
                    ST_LocateAlong({2}, cpa_point),
                1)
            )
        """
        self.cur.execute(sql_point.format(self.table, 'own_point', 'own_track'))
        self.cur.execute(sql_point.format(self.table, 'target_point', 'target_track'))

        self.add_column('cpa_time', 'TIMESTAMP', geometry=False)
        sql_time = """
            UPDATE {0}
            SET {1} = to_timestamp(cpa_point)
        """.format(self.table, 'cpa_time')
        self.cur.execute(sql_time)

        self.add_column('point_distance', 'FLOAT(4)', geometry=False)
        sql_distance = """
            UPDATE {0}
            SET {1} = ST_Distance(own_point::geography, target_point::geography)
        """.format(self.table, 'point_distance')
        self.cur.execute(sql_distance)
        self.conn.commit()


# -----------------------------------------------------------------------------
class ROT_Table(Postgres_Table):

    def __init__(self, conn, table):
        '''Connect to default database.'''
        super(ROT_Table, self).__init__(conn, table)
        self.cur = self.conn.cursor()

    def select_rot(self, min_rot):
        self.min_rot = min_rot
        sql = """
            CREATE TABLE {0} AS
            SELECT *
            FROM nais_points_10
            WHERE rot > {1}
        """.format(self.table, self.min_rot)
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
            'own_tss',
            'target_mmsi',
            'target_sectorid',
            'target_sog',
            'target_heading',
            'target_rot',
            'target_vesseltype',
            'target_tss',
            'azimuth_deg',
            'point_distance',
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
                n1.sog AS own_sog,
                n1.cog AS own_cog,
                n1.heading AS own_heading,
                n1.rot AS own_rot,
                n1.vesseltype AS own_vesseltype,
                n1.length AS own_length,
                n1.geom AS own_geom,
                n1.in_tss AS own_tss,
                n2.mmsi AS target_mmsi,
                n2.sectorid AS target_sectorid,
                n2.trackid AS target_trackid,
                n2.sog AS target_sog,
                n2.cog AS target_cog,
                n2.heading AS target_heading,
                n2.rot AS target_rot,
                n2.vesseltype AS target_vesseltype,
                n2.length AS target_length,
                n2.geom AS target_geom,
                n2.in_tss AS target_tss,
                ST_Distance(ST_Transform(n1.geom, 7260), ST_Transform(n2.geom, 7260)) AS point_distance,
                DEGREES(ST_Azimuth(n1.geom, n2.geom)) AS azimuth_deg
            FROM {1} n1 INNER JOIN {2} n2
            ON n1.sectorid = n2.sectorid
            WHERE n1.basedatetime = n2.basedatetime
            AND n1.mmsi != n2.mmsi
        """.format(self.table, self.input, self.input)
        self.cur.execute(sql)
        self.conn.commit()

    def near_points_dataframe(self, max_distance):
        '''Return dataframe of near points within max distance.'''
        cond = 'point_distance <= {0}'.format(max_distance)
        return self.table_dataframe(self.columnString, cond)

    def near_table(self, max_distance):
        '''Create near table in pandas using max_distance = 1nm.'''
        # 1nm = 1852 meters
        df = self.near_points_dataframe(max_distance)
        df['bearing'] = (df['azimuth_deg'] - df['own_heading']) % 360
        return df[self.columns].copy()

    def near_plot(self, max_distance, display_points):
        '''Plot the near points all in reference to own ship.'''
        self.df_near = self.near_table(max_distance).head(display_points)
        self.df_near = self.df_near[
            (self.df_near['own_sog'] >10) & (self.df_near['target_sog'] > 10)].copy()
        theta = np.array(self.df_near['bearing'])
        r = np.array(self.df_near['point_distance'])
        tss = np.array(self.df_near['bearing'])
        # colors = theta.apply(np.radians)
        colors = tss
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='polar')
        c = ax.scatter(theta, r, c=colors, cmap='hsv', alpha=0.75, s=1)
        # plt.legend(loc='upper left')
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
