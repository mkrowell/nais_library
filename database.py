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
from multiprocessing.dummy import Pool
import numpy as np
import os
from os.path import dirname, exists, join
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

        # file parameters
        self.root = tempfile.mkdtemp()
        self.nais_file = join(self.root, 'AIS_All.csv')

        # spatial parameters
        param_yaml = join(dirname(__file__), 'settings.yaml')
        with open(param_yaml, 'r') as stream:
            self.parameters = yaml.safe_load(stream)[self.city]

        self.zone = self.parameters['zone']
        self.lonMin = self.parameters['lonMin']
        self.lonMax = self.parameters['lonMax']
        self.latMin = self.parameters['latMin']
        self.latMax = self.parameters['latMax']
        self.srid = self.parameters['srid']

        # time parameters
        # self.months = [str(i).zfill(2) for i in range(1, 13)]
        self.months = ['01']

        # database parameters
        self.conn = psycopg2.connect(
            host='localhost',
            dbname='postgres',
            user='postgres',
            password=self.password)

        # Add functions to database
        sql = """
            CREATE OR REPLACE FUNCTION normalize_angle(
                angle FLOAT,
                range_start FLOAT,
                range_end FLOAT)
            RETURNS FLOAT AS $$
            BEGIN
                RETURN (angle - range_start) - FLOOR((angle - range_start)/(range_end - range_start))*(range_end - range_start) + range_start;
            END; $$
            LANGUAGE PLPGSQL;

            CREATE OR REPLACE FUNCTION angle_difference(
                angle1 FLOAT,
                angle2 FLOAT)
            RETURNS FLOAT AS $$
            BEGIN
                RETURN DEGREES(ATAN2(
                    SIN(RADIANS(angle1) - RADIANS(angle2)),
                    COS(RADIANS(angle1) - RADIANS(angle2))
                ));
            END; $$
            LANGUAGE PLPGSQL
        """
        self.conn.cursor().execute(sql)
        self.conn.commit()

        # Add SRID
        self.conn.cursor().execute(self.srid)
        self.conn.commit()

        # environment
        self.table_shore = Shapefile_Table(
            self.conn,
            'shore'
        )
        self.table_tss = Shapefile_Table(
            self.conn,
            'tss'
        )

        # eda
        self.table_eda = EDA_Table(
            self.conn,
            'eda_points_{0}'.format(self.city)
        )
        self.table_eda_tracks_mmsi = Tracks_Table(
            self.conn,
            'eda_tracks_mmsi_{0}'.format(self.city)
        )
        self.table_eda_tracks_trajectory = Tracks_Table(
            self.conn,
            'eda_tracks_trajectory_{0}'.format(self.city)
        )



        # self.table_points = Points_Table(
        #     self.conn,
        #     'points_{0}'.format(self.city)
        # )
        # self.table_cpas = CPA_Table(
        #     self.conn,
        #     'cpa_{0}'.format(self.city),
        #     self.table_points.table
        # )


        # self.table_tracks = Tracks_Table(
        #     self.conn,
        #     'tracks_{0}'.format(self.city)
        # )
        # self.table_cpa = CPA_Table(
        #     self.conn,
        #     'cpa_{0}'.format(self.city),
        #     self.table_tracks.table,
        #     self.table_shore.table
        # )

        # # Encounters
        # self.table_encounters = Encounters_Table(
        #     self.conn,
        #     'encounters_{0}_cog'.format(self.city),
        #     self.table_points.table,
        #     self.table_cpa.table
        # )
        #
        #






        # self.table_crossing = Crossing_Table(
        #     self.conn,
        #     'crossing_{0}'.format(self.city),
        #     self.table_analysis.table
        # )
        # self.table_overtaking = Overtaking_Table(
        #     self.conn,
        #     'overtaking_{0}'.format(self.city),
        #     self.table_analysis.table
        # )

    @property
    def nais_csvs(self):
        return glob(self.root + '\\AIS*.csv')

    # BUILD DATABASE -----------------------------------------------------------
    def build_tables(self):
        '''Build database of raw data.'''
        start = time.time()
        try:
            # Environmental
            self.build_shore()
            self.build_tss()

            # Points
            # self.build_nais_points()

            # Tracks
            # self.build_nais_tracks()

            # CPAsdf
            # self.build_nais_cpas()

            # Analysis
            # self.build_nais_encounters()
            # self.build_nais_analysis()

        except Exception as err:
            print(err)
            self.conn.rollback()
            self.conn.close()
        finally:
            shutil.rmtree(self.root)
            end = time.time()
            print('Elapsed Time: {0} minutes'.format((end-start)/60))

    def build_shore(self):
        '''Construct shoreline table.'''
        shore = Shoreline_Download(self.root)
        self.table_shore.create_table(filepath=shore.download_shoreline())

        # Transform to UTM 10 SRID
        self.table_shore.project_column('geom', 'MULTILINESTRING', 32610)

        # Keep only the relevant shore line
        self.table_shore.reduce_table('regions', '!=', self.parameters['region'])
        self.table_shore.add_index("idx_geom", "geom", type="gist")

    def build_tss(self):
        '''Construct TSS table.'''
        tss = TSS_Download(self.root)
        self.table_tss.create_table(filepath=tss.download_tss())

        # Transform to UTM 10 SRID
        self.table_shore.project_column('geom', 'MULTILINESTRING', 32610)

        # Keep only the relevant TSS
        self.table_tss.reduce_table('objl', '!=', self.parameters['tss'])
        self.table_tss.add_index("idx_geom", "geom", type="gist")

    def build_nais_eda(self):
        '''Build nais exploratory data analysis table.'''
        # Download and process raw AIS data from MarineCadastre.gov
        if not exists(self.nais_file):
            raw = NAIS_Download(self.root, self.city, self.year)
            for month in self.months:
                raw.download_nais(month)
            raw.clean_up()
            raw.preprocess_eda()

        # Create table
        self.table_eda.drop_table()
        self.table_eda.create_table()
        self.table_eda.copy_data(self.nais_file)
        self.table_eda.add_local_time()

        # Add postgis point and project to UTM 10 SRID
        self.table_eda.add_geometry()
        self.table_eda.project_column('geom', 'POINTM', 32610)
        self.table_eda.add_index("idx_geom", "geom", type="gist")

        # Make tracks on MMSI only and on MMSI, Trajectory
        points = self.table_eda.table
        self.table_eda_tracks_mmsi.drop_table()
        self.table_eda_tracks_mmsi.convert_to_tracks(points, groupby='mmsi, type')
        self.table_eda_tracks_trajectory.drop_table()
        self.table_eda_tracks_trajectory.convert_to_tracks(points, groupby='mmsi, trajectory, type')



    def build_nais_points(self):
        '''Build nais points table.'''
        # Download and process raw AIS data from MarineCadastre.gov
        print('Constructing nais_points table...')
        if not exists(self.nais_file):
            raw = NAIS_Download(self.root, self.city, self.year)
            for month in self.months:
                raw.download_nais(month)
            raw.clean_up()
            raw.preprocess_nais()

        # Create table
        self.table_points.drop_table()
        self.table_points.create_table()
        self.table_points.copy_data(self.nais_file)

        # Add postgis point and project to UTM 10 SRID
        self.table_points.add_geometry()
        self.table_points.project_column('geom', 'POINTM', 32610)
        self.table_points.add_index("idx_geom", "geom")
        self.table_points.add_tss(self.table_tss.table)

        # Add time index for join
        self.table_points.add_index("idx_time", "datetime")

    def build_nais_interactions(self):
        '''Join points to points to get CPA.'''
        self.table_cpas.drop_table()

        # Join points to points and select closest CPAs
        self.table_cpas.points_points()
        self.table_cpas.add_index("idx_distance", "cpa_distance")
        self.table_cpas.reduce_table('cpa_distance', '>=', 0.5*1852)

        # Add rank to duplicates
        self.table_cpas.add_index("idx_mmsi", "mmsi1")
        self.table_cpas.add_duplicate_rank()

        # Select the 20 points around the CPA
        self.table_cpas.cpa_range()
        self.table_cpas.cpa_attributes()

        # Add encounter type
        self.table_cpas.encounter_type()

    def build_nais_tracks(self):
        '''Create tracks table from points table.'''
        print('Constructing nais_tracks table...')
        self.table_tracks.drop_table()
        self.table_tracks.convert_to_tracks(self.table_points.table)

        self.table_tracks.add_index("idx_gist_period", "period", type="gist")
        self.table_tracks.add_index("idx_duration", "duration")

        self.table_tracks.reduce_table('duration', '<=', '00:30:00')

    def build_nais_cpas(self):
        '''Create table to generate pair-wise cpa.'''
        self.table_cpa.drop_table()

        # Join tracks pairwise
        self.table_cpa.tracks_tracks()
        self.table_cpa.remove_null('cpa_epoch')

        # Add CPA attributes
        self.table_cpa.cpa_time()
        self.table_cpa.cpa_points()
        self.table_cpa.cpa_distance()

        self.table_cpa.add_index("idx_distance", "point_distance")
        self.table_cpa.reduce_table('point_distance','>',  3*1852)

        # Remove cpas that cross the shore
        self.table_cpa.cpa_line()
        self.table_cpa.delete_shore_cross()

        # Add rank to duplicate interactions
        self.table_cpa.add_duplicate_rank()

    def build_nais_encounters(self):
        '''Add point data to cpa instances.'''
        self.table_encounters.drop_table()

        # Add point data to CPA
        self.table_encounters.cpa_points()

        # Get encounter type
        self.table_encounters.encounter_type()
        self.table_encounters.check_nearby()

        # Add encounter info
        self.table_encounters.giveway_info()
        self.table_encounters.dcpa()
        self.table_encounters.tcpa()








#         # plot every encounter cog v time, sog v time, step_cog v tcpa
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='polar')
# ax.set_rlabel_position(135)
# ax.set_theta_zero_location('N')
# ax.set_theta_direction(-1)
# c = ax.scatter(np.radians(overtaking['bearing12']), overtaking['distance'], c=np.radians(overtaking['bearing12']), cmap='hsv', alpha=0.75)
#

# ------------------------------------------------------------------------------
# TABLES
# ------------------------------------------------------------------------------
class Postgres_Table(object):

    def __init__(self, conn, table):
        '''Connect to database.'''
        # Connect to the database and create a cursor
        self.conn = conn
        self.cur = self.conn.cursor()
        self.table = table

        # Enable PostGIS extension
        self.cur.execute("CREATE EXTENSION IF NOT EXISTS postgis")
        self.conn.commit()
        register(self.conn)

        # Set timezone to UTC
        self.cur.execute("SET timezone = 'UTC'")
        self.conn.commit()

    def drop_table(self, table=None):
        '''Drop the table if it exists.'''
        if table is None:
            table = self.table
        sql = "DROP TABLE IF EXISTS {0}".format(table)

        print("Dropping table {0}...".format(table))
        self.cur.execute(sql)
        self.conn.commit()

    def drop_column(self, column):
        '''Drop the column if it exists.'''
        sql = """
            ALTER TABLE {0}
            DROP COLUMN IF EXISTS {1}
        """.format(self.table, column)

        print("Dropping column {0}...".format(column))
        self.cur.execute(sql)
        self.conn.commit()

    def copy_data(self, csv_file):
        '''Copy data into table.'''
        with open(csv_file, 'r') as f:
            print('Copying {0} to database...'.format(csv_file))
            self.cur.copy_from(f, self.table, sep=',')
            self.conn.commit()

    def create_table(self, filepath=None):
        '''Create given table.'''
        print('Constructing {0}...'.format(self.table))
        if filepath:
            cmd = "shp2pgsql -s 4326 -d {0} {1} | psql -d postgres -U postgres -q".format(filepath, self.table)
            subprocess.call(cmd, shell=True)
        else:
            sql = """
                CREATE TABLE IF NOT EXISTS {0} ({1})
            """.format(self.table, self.columns)
            self.cur.execute(sql)
            self.conn.commit()

    def add_index(self, name, field, type=None):
        '''Add index to table using the given column.'''
        print('Adding index on {0}...'.format(field))
        if type is None:
            sql = """
                CREATE INDEX IF NOT EXISTS {0}
                ON {1} ({2})
            """
        if type == 'gist':
            sql = """
                CREATE INDEX IF NOT EXISTS {0}
                ON {1} USING GiST ({2})
            """
        self.cur.execute(sql.format(name, self.table, field))
        self.conn.commit()

    def add_column(self, name, datatype=None, geometry=False, default=None, srid=4326):
        '''Add column with datatype to the table.'''
        print('Adding {0} ({1}) to {2}...'.format(name, datatype, self.table))
        sql_alter = """
            ALTER TABLE {0}
            ADD COLUMN IF NOT EXISTS {1}
        """.format(self.table, name)

        # Handle geometry types
        sql_type = """ {0} """
        if geometry:
            sql_type = """ geometry({0}, {1}) """
        sql = sql_alter + sql_type.format(datatype, srid)

        # Handle default data types
        if default:
            sql_default = """DEFAULT {0}"""
            sql = sql + sql_default.format(default)

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

    def project_column(self, column, datatype, new_srid):
        '''Transform projection.'''
        print('Projecting {0} from 4326 to {1}...'.format(column, new_srid))
        sql = """
            ALTER TABLE {table}
            ALTER COLUMN {column}
            TYPE Geometry({datatype}, {new_srid})
            USING ST_Transform({column}, {new_srid});
        """.format(table=self.table, column=column, datatype=datatype, new_srid=new_srid)
        self.cur.execute(sql)
        self.conn.commit()

    def add_local_time(self, input_col="datetime", timezone="america/los_angeles"):
        '''Add local time.'''
        name = 'datetime_local'
        self.add_column(name, datatype='timestamptz')
        sql = """
            UPDATE {table}
            SET {newCol} = {inputCol} at time zone 'utc' at time zone '{zone}'
        """.format(table=self.table, newCol=name, inputCol=input_col, zone=timezone)
        self.cur.execute(sql)
        self.conn.commit()

    def reduce_table(self, column, relationship, value):
        '''Drop rows on one NOT EQUAL TO condition.'''
        print('Dropping {0} {1} {2} from {3}...'.format(
            column,
            relationship,
            value,
            self.table)
        )
        if isinstance(value, str):
            sql_delete = "DELETE FROM {0} WHERE {1} {2} '{3}'"
        else:
            sql_delete = "DELETE FROM {0} WHERE {1} {2} {3}"
        sql = sql_delete.format(self.table, column, relationship, value)
        self.cur.execute(sql)
        self.conn.commit()

    def remove_null(self, col):
        print('Deleting null rows from {0}...'.format(self.table))
        sql = """
            DELETE FROM {0} WHERE {1} IS NULL
        """.format(self.table, col)
        self.cur.execute(sql)
        self.conn.commit()

    def table_dataframe(self, table=None, select_col=None, where_cond=None):
        '''Return dataframe.'''
        if table is None:
            table = self.table
        if select_col is None:
            select_col = '*'
        sql = """
            SELECT {0}
            FROM {1}
        """.format(select_col, table)

        if where_cond is not None:
            sql = sql + """ WHERE {0} """.format(where_cond)

        self.cur.execute(sql)
        column_names = [desc[0] for desc in self.cur.description]
        return pd.DataFrame(self.cur.fetchall(), columns=column_names)

class Shapefile_Table(Postgres_Table):

    def __init__(self, conn, table):
        '''Connect to default database.'''
        super(Shapefile_Table, self).__init__(conn, table)
        self.cur = self.conn.cursor()

class EDA_Table(Postgres_Table):

    def __init__(self, conn, table):
        '''Connect to default database.'''
        super(EDA_Table, self).__init__(conn, table)
        self.cur = self.conn.cursor()
        self.columns = """
            MMSI integer NOT NULL,
            DateTime timestamptz NOT NULL,
            Trajectory integer NOT NULL,
            LAT float8 NOT NULL,
            LON float8 NOT NULL,
            SOG float(4) NOT NULL,
            COG float(4) NOT NULL,
            VesselType varchar(64)
        """

    def add_geometry(self):
        '''Add PostGIS PointM geometry to the database and make it index.'''
        self.add_column('geom', datatype='POINTM', geometry=True)
        self.add_point('geom', 'lon', 'lat', "date_part('epoch', datetime)")

class Tracks_Table(Postgres_Table):

    def __init__(self, conn, table):
        '''Connect to default database.'''
        super(Tracks_Table, self).__init__(conn, table)
        self.cur = self.conn.cursor()

    def convert_to_tracks(self, points, groupby):
        '''Add LINESTRING for each MMSI, TrackID.'''
        print('Creating tracks from points...')
        sql = """
            CREATE TABLE {table} AS
            SELECT
                mmsi,
                trajectory,
                vesseltype,
                max(datetime) - min(datetime) AS duration,
                tsrange(min(datetime), max(datetime)) AS period,
                ST_MakeLine(geom ORDER BY datetime) AS track
            FROM {points}
            GROUP BY {group}
            HAVING COUNT(*) > 2
            """.format(table=self.table, points=points, group=groupby)
        self.cur.execute(sql)
        self.conn.commit()






class Points_Table(Postgres_Table):

    def __init__(self, conn, table):
        '''Connect to default database.'''
        super(Points_Table, self).__init__(conn, table)
        self.cur = self.conn.cursor()
        self.columns = """
            MMSI integer NOT NULL,
            DateTime timestamptz NOT NULL,
            Track integer NOT NULL,
            Step_COG_Degrees float(4) NOT NULL,
            Step_COG_Radians float(4) NOT NULL,
            COG_Cosine float(4) NOT NULL,
            Step_Acceleration float(4) NOT NULL,
            LAT float8 NOT NULL,
            LON float8 NOT NULL,
            SOG float(4) NOT NULL,
            COG float(4) NOT NULL,
            Point_COG float(4) NOT NULL,
            Heading float(4) NOT NULL,
            VesselName varchar(32),
            VesselType varchar(64) NOT NULL,
            Status varchar(64),
            Length float(4),
            Width float(4)
        """

    def add_geometry(self):
        '''Add PostGIS PointM geometry to the database and make it index.'''
        self.add_column('geom', datatype='POINTM', geometry=True)
        self.add_point('geom', 'lon', 'lat', "date_part('epoch', datetime_local)")

    def add_tss(self, tss):
        '''Add column marking whether the point is in the TSS.'''
        name = 'in_tss'
        self.add_column(name, datatype='boolean', default='FALSE')
        sql = """
            UPDATE {table}
            SET {col} = TRUE
            FROM (
                SELECT points.lat, points.lon
                FROM {table} AS points
                RIGHT JOIN {tssTable} AS polygons
                ON ST_Contains(polygons.geom, points.geom)
            ) as tssTest
            WHERE {table}.lat=tssTest.lat
            AND {table}.lon=tssTest.lon
        """.format(table=self.table, col=name, tssTable=tss)
        self.cur.execute(sql)
        self.conn.commit()

# class Tracks_Table(Postgres_Table):
#
#     def __init__(self, conn, table):
#         '''Connect to default database.'''
#         super(Tracks_Table, self).__init__(conn, table)
#         self.cur = self.conn.cursor()
#
#     def convert_to_tracks(self, points):
#         '''Add LINESTRING for each MMSI, TrackID.'''
#         print('Creating tracks from points...')
#         sql = """
#             CREATE TABLE {0} AS
#             SELECT
#                 mmsi,
#                 track AS id,
#                 vesseltype AS type,
#                 max(datetime) - min(datetime) AS duration,
#                 tsrange(min(datetime), max(datetime)) AS period,
#                 ST_MakeLine(geom ORDER BY datetime) AS track
#             FROM {1}
#             GROUP BY mmsi, id, type
#             HAVING COUNT(*) > 2
#             """.format(self.table, points)
#         self.cur.execute(sql)
#         self.conn.commit()
#


# class CPA_Table(Postgres_Table):
#
#     def __init__(self, conn, table, input_table):
#         '''Connect to default database.'''
#         super(CPA_Table, self).__init__(conn, table)
#         self.cur = self.conn.cursor()
#         self.input = input_table
#
#     def points_points(self):
#         '''Join points to itself to get closest points.'''
#         print('Joining points to points to find CPAs...')
#         sql = """
#             CREATE TABLE {table} AS
#             SELECT
#                 p.basedatetime as datetime,
#                 p.mmsi AS mmsi1,
#                 p.track AS track1,
#                 p.step_cog_degrees AS step_cog1,
#                 p.cog_cosine AS cog_cos1,
#                 p.step_acceleration AS accel1,
#                 p.point_cog AS cog1,
#                 p.heading AS heading1,
#                 p.sog AS sog1,
#                 p.vesseltype AS type1,
#                 p.length AS length1,
#                 p.in_tss AS point_tss1,
#                 p.geom AS point1,
#                 p2.mmsi AS mmsi2,
#                 p2.track AS track2,
#                 p2.step_cog_degrees AS step_cog2,
#                 p2.cog_cosine AS cog_cos2,
#                 p2.step_acceleration AS accel2,
#                 p2.point_cog AS cog2,
#                 p2.heading AS heading2,
#                 p2.sog AS sog2,
#                 p2.vesseltype AS type2,
#                 p2.length AS length2,
#                 p2.in_tss AS point_tss2,
#                 p2.geom AS point2,
#                 ST_Distance(p.geom, p2.geom)::int AS distance,
#                 min(ST_Distance(p.geom, p2.geom)::int) OVER (PARTITION BY p.mmsi, p.track, p2.mmsi, p2.track) AS cpa_distance,
#                 normalize_angle(angle_difference(DEGREES(ST_Azimuth(p.geom, p2.geom)), p.heading),0,360) AS bearing12,
#                 ROW_NUMBER() OVER (PARTITION BY p.mmsi, p.track, p2.mmsi, p2.track ORDER BY p2.basedatetime ASC) as rownum
#             FROM {points} AS p LEFT JOIN {points} AS p2
#             ON p.basedatetime = p2.basedatetime
#             AND p.mmsi != p2.mmsi
#         """.format(table=self.table, points=self.input)
#         self.cur.execute(sql)
#         self.conn.commit()
#
#     def add_duplicate_rank(self):
#         '''Rank duplicate interactions.'''
#         rank = 'rank'
#         self.add_column(rank, 'integer')
#
#         sql = """
#             UPDATE {table}
#             SET {rank} = CASE
#             WHEN mmsi1::int > mmsi2::int THEN 1 ELSE 2 END
#         """.format(table=self.table, rank=rank)
#         self.cur.execute(sql)
#         self.conn.commit()
#
#     def cpa_range(self, buffer=10):
#         '''Keep only the 20 points around the CPA.'''
#         sql = """
#             WITH current AS (
#                 SELECT mmsi1, track1, mmsi2, track2, rownum
#                 FROM {table}
#                 WHERE distance = cpa_distance
#             )
#
#             DELETE FROM {table}
#             USING current
#             WHERE {table}.mmsi1 = current.mmsi1
#             AND {table}.track1 = current.track1
#             AND {table}.mmsi2 = current.mmsi2
#             AND {table}.track2 = current.track2
#             AND {table}.rownum BETWEEN current.rownum - {buffer} AND current.rownum + {buffer}
#         """.format(table=self.table, buffer=buffer)
#         self.cur.execute(sql)
#         self.conn.commit()
#
#     def cpa_attributes(self, buffer=10):
#         '''Keep only the 20 points around the CPA.'''
#         time = 'cpa_time'
#         self.add_column(time, 'timestamp')
#
#         point1 = 'cpa_point1'
#         self.add_column(point1, datatype='pointm', geometry=True)
#
#         point2 = 'cpa_point2'
#         self.add_column(point2,  datatype='pointm', geometry=True)
#         sql = """
#             WITH current AS (
#                 SELECT mmsi1, track1, mmsi2, track2, rownum
#                 FROM {table}
#                 WHERE distance = cpa_distance
#             )
#             UPDATE {table}
#             SET
#                 {time} = {table}.datetime,
#                 {p1} = {table}.point1,
#                 {p2} = {table}.point2
#             FROM current
#             WHERE {table}.mmsi1 = current.mmsi1
#             AND {table}.track1 = current.track1
#             AND {table}.mmsi2 = current.mmsi2
#             AND {table}.track2 = current.track2
#             AND {table}.rownum = current.rownum
#         """.format(table=self.table, time=time, p1=point1, p2=point2)
#         self.cur.execute(sql)
#         self.conn.commit()
#
#     def encounter_type(self):
#         '''Add type of interaction.'''
#         conType = 'encounter'
#         self.add_column(conType, datatype='varchar')
#
#         sql = """
#             WITH first AS (
#                 SELECT *
#                 FROM {table}
#                 WHERE (mmsi1, track1, mmsi2, track2) IN (
#                     SELECT mmsi1, track1, mmsi2, track2
#                     FROM (
#                         SELECT
#                             mmsi1,
#                             track1,
#                             mmsi2,
#                             track2,
#                             ROW_NUMBER() OVER(PARTITION BY mmsi1, track1, mmsi2, track2 ORDER BY datetime ASC) as rk
#                         FROM {table}
#                     ) AS subquery
#                 WHERE rk = 1
#                 )
#             )
#
#             UPDATE {table}
#             SET {type} = CASE
#                 WHEN @first.cogdiff12 BETWEEN 165 AND 195 THEN 'head-on'
#                 WHEN @first.cogdiff12 < 15 OR @first.cogdiff12 > 345 THEN 'overtaking'
#                 ELSE 'crossing'
#                 END
#             FROM first
#             WHERE first.mmsi1 = {table}.mmsi1
#             AND first.track1 = {table}.track1
#             AND first.mmsi2 = {table}.mmsi2
#             AND first.track2 = {table}.track2
#         """.format(table=self.table, type=conType)
#         self.cur.execute(sql)
#         self.conn.commit()
#
#     def check_nearby(self):
#         '''Delete encounters that are just nearby and not really encounter.'''
#         print('Deleting near ships with no encounter...')
#         sql = """
#             DELETE FROM {0}
#             WHERE (
#                 (encounter = 'head-on' OR encounter = 'overtaking')
#                 AND (
#                     90 NOT BETWEEN min(bearing12::int) and max(bearing12::int) OR
#                     270 NOT BETWEEN min(bearing12::int) and max(bearing12::int)
#             ) OR (
#                 (encounter = 'crossing')
#                 AND (
#                     0 NOT BETWEEN min(bearing12::int)-5 and min(bearing12::int) +5 OR
#                     360 NOT BETWEEN max(bearing12::int)-5 and max(bearing12::int) +5 OR
#                     180 NOT BETWEEN min(bearing12::int) and max(bearing12::int)
#             )
#         """.format(self.table)
#         self.cur.execute(sql)
#         self.conn.commit()
#



    # int4range(
    #      min(normalize_angle(angle_difference(DEGREES(ST_Azimuth(p.geom, p2.geom)), p.heading),0,360)) OVER (PARTITION BY p.mmsi, p.track, p2.mmsi, p2.track)::int,
    #      max(normalize_angle(angle_difference(DEGREES(ST_Azimuth(p.geom, p2.geom)), p.heading),0,360)) OVER (PARTITION BY p.mmsi, p.track, p2.mmsi, p2.track)::int) AS brange,







class CPA_Table(Postgres_Table):

    def __init__(self, conn, table, input_table, shore_table):
        '''Connect to default database.'''
        super(CPA_Table, self).__init__(conn, table)
        self.cur = self.conn.cursor()
        self.input = input_table
        self.shore = shore_table

    def tracks_tracks(self):
        '''Make pairs of tracks that happen in same time interval.'''
        print('Joining tracks with itself to make interaction table...')
        sql = """
            CREATE TABLE {table} AS
            SELECT
                n1.mmsi AS mmsi1,
                n1.id AS id1,
                n1.track AS track1,
                n2.mmsi AS mmsi2,
                n2.id AS id2,
                n2.track AS track2,
                ST_ClosestPointOfApproach(n1.track, n2.track) AS cpa_epoch,
                ST_DistanceCPA(n1.track, n2.track) AS cpa_distance
            FROM {input} n1 LEFT JOIN {input} n2
            ON n1.period && n2.period
            AND n1.mmsi != n2.mmsi
        """.format(table=self.table, input=self.input)
        self.cur.execute(sql)
        self.conn.commit()

    def cpa_points(self):
        '''Add track points.'''
        self.add_column('cpa_point1', 'POINTM', geometry=True, srid=32610)
        self.add_column('cpa_point2', 'POINTM', geometry=True, srid=32610)
        sql = """
            UPDATE {0}
            SET
                cpa_point1 = ST_Force3DM(
                    ST_GeometryN(
                        ST_LocateAlong(track1, cpa_epoch),
                        1
                    )
                ),
                cpa_point2 = ST_Force3DM(
                    ST_GeometryN(
                        ST_LocateAlong(track2, cpa_epoch),
                        1
                    )
                )
        """
        self.cur.execute(sql)
        self.conn.commit()

    def cpa_time(self):
        '''Add track time.'''
        self.add_column('cpa_time', 'TIMESTAMP', geometry=False)
        sql = """
            UPDATE {0}
            SET {1} = to_timestamp(cpa_epoch)
        """.format(self.table, 'cpa_time')
        self.cur.execute(sql)
        self.conn.commit()

    def cpa_line(self):
        '''Add line between CPA points.'''
        self.add_column('cpa_line', 'LINESTRINGM', geometry=True, srid=32610)
        sql = """
            UPDATE {0}
            SET {1} = ST_MakeLine(cpa_point1, cpa_point2)
        """.format(self.table, 'cpa_line')
        self.cur.execute(sql)
        self.conn.commit()

    def delete_shore_cross(self):
        '''Delete CPAs that line on shore.'''
        print('Deleting shore intersects from {0}...'.format(self.table))
        sql = """
            DELETE FROM {table} c
            USING {shore} s
            WHERE ST_Intersects(c.cpa_line, s.geom)
        """.format(table=self.table, shore=self.shore)
        self.cur.execute(sql)
        self.conn.commit()



class Encounters_Table(Postgres_Table):

    def __init__(self, conn, table, input_points, input_cpa):
        '''Connect to default database.'''
        super(Encounters_Table, self).__init__(conn, table)
        self.cur = self.conn.cursor()
        self.points = input_points
        self.cpa = input_cpa

    def cpa_points(self):
        '''Make pairs of tracks that happen in same time interval.'''
        print('Joining points with cpas to make encounter table...')
        sql = """
            CREATE TABLE {table} AS
            SELECT
                c.mmsi1,
                c.id1 AS track1,
                c.mmsi2,
                c.id2 AS track2,
                c.rank,
                p.basedatetime AS datetime,
                c.cpa_time,
                c.point_distance,
                c.cpa_point1,
                c.cpa_point2,
                p.step_cog_degrees AS step_cog1,
                p.cog_cosine AS cog_cos1,
                p.step_acceleration AS accel1,
                p.point_cog AS cog1,
                p.heading AS heading1,
                p.sog AS sog1,
                p.vesseltype AS type1,
                p.length AS length1,
                p.in_tss AS point_tss1,
                p.geom AS point1,
                p2.step_cog_degrees AS step_cog2,
                p2.cog_cosine AS cog_cos2,
                p2.step_acceleration AS accel2,
                p2.point_cog AS cog2,
                p2.heading AS heading2,
                p2.sog AS sog2,
                p2.vesseltype AS type2,
                p2.length AS length2,
                p2.in_tss AS point_tss2,
                p2.geom AS point2,
                ST_Distance(p.geom, p2.geom) AS distance,
                normalize_angle(angle_difference(DEGREES(ST_Azimuth(p.geom, p2.geom)), p.heading),0,360) AS bearing12,
                int4range(
                    min(normalize_angle(angle_difference(DEGREES(ST_Azimuth(p.geom, p2.geom)), p.heading),0,360)) OVER (PARTITION BY c.mmsi1, c.track1, c.mmsi2, c.track2)::int,
                    max(normalize_angle(angle_difference(DEGREES(ST_Azimuth(p.geom, p2.geom)), p.heading),0,360)) OVER (PARTITION BY c.mmsi1, c.track1, c.mmsi2, c.track2)::int) AS brange,
                angle_difference(p.point_cog, p2.point_cog) AS cogdiff12,
                AVG(p.cog_cosine) OVER (PARTITION BY c.mmsi1, c.id1, c.mmsi1, c.id2) AS sinuosity1,
                AVG(p2.cog_cosine) OVER (PARTITION BY c.mmsi1, c.id1, c.mmsi1, c.id2) AS sinuosity2,
                min(p2.basedatetime) OVER (PARTITION BY c.mmsi1, c.id1, c.mmsi2, c.id2) AS period_start,
                max(p2.basedatetime) OVER (PARTITION BY c.mmsi1, c.id1, c.mmsi2, c.id2) AS period_end
            FROM {cpa} c
            LEFT JOIN {points} p
                ON p.basedatetime between c.cpa_time - INTERVAL '10 minutes' AND c.cpa_time + INTERVAL '10 minutes'
                AND p.mmsi = c.mmsi1
            LEFT JOIN {points} p2
                ON p2.mmsi = c.mmsi2
                AND p2.basedatetime = p.basedatetime
            WHERE point_distance <= 2*1852
            AND p.geom IS NOT NULL
            AND p2.geom IS NOT NULL
        """.format(table=self.table, cpa=self.cpa, points=self.points)
        self.cur.execute(sql)
        self.conn.commit()

    def encounter_type(self):
        '''Add type of interaction.'''
        conType = 'encounter'
        self.add_column(conType, datatype='varchar')

        sql = """
            WITH first AS (
                SELECT *
                FROM {table}
                WHERE (mmsi1, track1, mmsi2, track2) IN (
                    SELECT mmsi1, track1, mmsi2, track2
                    FROM (
                        SELECT
                            mmsi1,
                            track1,
                            mmsi2,
                            track2,
                            ROW_NUMBER() OVER(PARTITION BY mmsi1, track1, mmsi2, track2 ORDER BY datetime ASC) as rk
                        FROM {table}
                    ) AS subquery
                WHERE rk = 1
                )
            )

            UPDATE {table}
            SET {type} = CASE
                WHEN @first.cogdiff12 BETWEEN 165 AND 195 THEN 'head-on'
                WHEN @first.cogdiff12 < 15 OR @first.cogdiff12 > 345 THEN 'overtaking'
                ELSE 'crossing'
                END
            FROM first
            WHERE first.mmsi1 = {table}.mmsi1
            AND first.track1 = {table}.track1
            AND first.mmsi2 = {table}.mmsi2
            AND first.track2 = {table}.track2
        """.format(table=self.table, type=conType)
        self.cur.execute(sql)
        self.conn.commit()



    def giveway_info(self):
        '''Add type of interaction.'''
        print('Adding gvieway info to {0}'.format(self.table))
        vessel = 'give_way'
        self.add_column(vessel, datatype='integer')

        sql = """
            WITH first AS (
                SELECT *
                FROM {table}
                WHERE (mmsi1, track1, mmsi2, track2) IN (
                    SELECT mmsi1, track1, mmsi2, track2
                    FROM (
                        SELECT
                            mmsi1,
                            track1,
                            mmsi2,
                            track2,
                            ROW_NUMBER() OVER(PARTITION BY mmsi1, track1, mmsi2, track2 ORDER BY datetime ASC) as rk
                        FROM {table}
                    ) AS subquery
                WHERE rk = 1
                )
            )

            UPDATE {table}
            SET
                {give} = CASE
                    WHEN {table}.encounter = 'overtaking' THEN CASE
                        WHEN first.bearing12 BETWEEN 90 AND 270 THEN 0
                        ELSE 1
                        END
                    WHEN {table}.encounter = 'crossing' THEN CASE
                        WHEN first.bearing12 BETWEEN 0 AND 112.5 THEN 1
                        ELSE 0
                        END
                    WHEN {table}.encounter = 'head-on' THEN 1
                    END
            FROM first
            WHERE first.mmsi1 = {table}.mmsi1
            AND first.track1 = {table}.track1
            AND first.mmsi2 = {table}.mmsi2
            AND first.track2 = {table}.track2
        """.format(table=self.table, give=vessel)
        self.cur.execute(sql)
        self.conn.commit()

    def dcpa(self):
        '''Add distance to CPA.'''
        name1 = 'dcpa1'
        name2 = 'dcpa2'

        self.add_column(name1, datatype='float(4)')
        self.add_column(name2, datatype='float(4)')

        sql = """
            UPDATE {0}
            SET {1} = ST_Distance({2}, {3})
        """
        self.cur.execute(sql.format(self.table, name1, 'point1', 'cpa_point1'))
        self.cur.execute(sql.format(self.table, name2, 'point2', 'cpa_point2'))
        self.conn.commit()

    def tcpa(self):
        '''Add time to CPA.'''
        name = 'tcpa'
        self.add_column(name, datatype='float(4)')
        sql = """
            UPDATE {0}
            SET {1} = EXTRACT(MINUTE FROM (datetime::timestamp - cpa_time::timestamp))
        """.format(self.table, name)
        self.cur.execute(sql)
        self.conn.commit()

    def time_range(self):
        '''Add time period of interaction.'''
        print('Adding time period to {0}'.format(self.table))
        period = 'period'
        duration = 'duration'

        self.add_column(period, datatype='tsrange')
        self.add_column(duration, datatype='time')

        sql = """
            UPDATE {table}
            SET
                {period} = tsrange(period_start, period_end),
                {duration} = period_end - period_start
        """.format(table=self.table, period=period, duration=duration)
        self.cur.execute(sql)
        self.conn.commit()

    def traffic(self):
        '''Add the number of other vessel's the ship is interacting with.'''
        name = 'traffic'
        self.add_column(name, datatype='integer')
        sql = """
            WITH count AS (
                SELECT
                    mmsi1,
                    track1,
                    period,
                    COUNT (DISTINCT mmsi2) AS traffic
                FROM {table}
                GROUP BY mmsi1, track1, period
            )

            UPDATE {table}
            SET {col} = count.traffic
            FROM count
            WHERE {table}.mmsi1 = count.mmsi1
            AND {table}.track1 = count.track1
            AND {table}.period && count.period
        """.format(table=self.table, col=name)
        self.cur.execute(sql)
        self.conn.commit()

    def plot_encounters(self):

        df = self.table.table_dataframe()
        for name, group in df.groupby(['mmsi1','track1','mmsi2','track2']):
            name = 'Plot_{0}_{1}_{2}_{3}.png'.format()
            fig = plt.figure(figsize=(12, 4), dpi=None, facecolor='white')
            fig.suptitle('Track Comparison', fontsize=12, fontweight='bold', x=0.5, y =1.01)
            plt.title(' => '.join([str(i) for i in name]), fontsize=10, loc='center')
            # plt.yticks([])

# ax1 = fig.add_subplot(111)
# ax1.set_ylabel('COG')
# plt.plot('tcpa', 'cog1', data=temp, marker='o', markerfacecolor='blue', markersize=5)
# plt.plot('tcpa', 'cog2', data=temp, marker='o', color='red', linewidth=2)
# plt.legend()
# ax1.grid(b=True, which='major', color='grey', linestyle='dotted')
#
# ax2 = fig.add_subplot(211)
# ax2.set_ylabel('SOG')
# plt.plot('tcpa', 'sog1', data=temp, marker='o', markerfacecolor='blue', markersize=5)
# plt.plot('tcpa', 'sog2', data=temp, marker='o', color='red', linewidth=2)
# plt.legend()
# ax2.grid(b=True, which='major', color='grey', linestyle='dotted')
#
# ax3 = fig.add_subplot(311)
# ax3.set_ylabel('Distance to CPA')
# plt.plot('tcpa', 'dcpa1', data=temp, marker='o', markerfacecolor='blue', markersize=5)
# plt.plot('tcpa', 'dcpa2', data=temp, marker='o', color='red', linewidth=2)
# plt.legend()
# ax3.grid(b=True, which='major', color='grey', linestyle='dotted')
#
# ax4 = fig.add_subplot(411)
# ax4.set_ylabel('Bearing from Ship 1 to Ship 2')
# plt.plot('tcpa', 'bearing12', data=temp, marker='o', markerfacecolor='black', markersize=5)
# plt.legend()
# ax4.grid(b=True, which='major', color='grey', linestyle='dotted')
#
#             fig.savefig(name, format="png", dpi=100, bbox_inches='tight', pad_inches=0.4)








# class Crossing_Table(Postgres_Table):
#
#     def __init__(self, conn, table, input_analysis):
#         '''Connect to default database.'''
#         super(Crossing_Table, self).__init__(conn, table)
#         self.cur = self.conn.cursor()
#         self.input = input_analysis
#         self.colregs = '{0}_colregs'.format(self.table)
#         self.others = '{0}_others'.format(self.table)
#
#     def select_type(self):
#         '''Select only the crossing interactions.'''
#         print('Selecting crossing interactions...')
#         sql = """
#             CREATE TABLE {0} AS
#             SELECT
#                 *,
#                 avg(stand_on_cog_cos) OVER (PARTITION BY mmsi1, track1, mmsi2, track2) AS avg_cog_cos,
#                 avg(stand_on_accel) OVER (PARTITION BY mmsi1, track1, mmsi2, track2) AS avg_accel
#             FROM {1}
#             WHERE encounter = 'crossing'
#         """.format(self.table, self.input)
#         self.cur.execute(sql)
#         self.conn.commit()
#
#     def separate_colregs(self):
#         '''Create table of colregs compliant interactions.'''
#         print('Selecting colreg compliant interactions...')
#         sql_colregs = """
#             CREATE TABLE {0} AS
#             SELECT *
#             FROM {1}
#             WHERE avg_cog_cos >= 0.999
#             AND avg_accel <= abs(10)
#         """.format(self.colregs, self.table)
#         self.cur.execute(sql_colregs)
#         self.conn.commit()
#
#         print('Selecting colreg compliant interactions...')
#         sql_others = """
#             CREATE TABLE {0} AS
#             SELECT *
#             FROM {1}
#             WHERE avg_cog_cos < 0.999
#             AND avg_accel > abs(10)
#         """.format(self.others, self.table)
#         self.cur.execute(sql_others)
#         self.conn.commit()
#
# class Overtaking_Table(Postgres_Table):
#
#     def __init__(self, conn, table, input_analysis):
#         '''Connect to default database.'''
#         super(Overtaking_Table, self).__init__(conn, table)
#         self.cur = self.conn.cursor()
#         self.input = input_analysis
#         self.colregs = '{0}_colregs'.format(self.table)
#         self.others = '{0}_others'.format(self.table)
#
#     def select_type(self):
#         '''Select only the overtaking interactions.'''
#         print('Selecting overtaking interactions...')
#         sql = """
#             CREATE TABLE {0} AS
#             SELECT
#                 *,
#                 avg(stand_on_cog_cos) OVER (PARTITION BY mmsi1, track1, mmsi2, track2) AS avg_cog_cos,
#                 avg(stand_on_accel) OVER (PARTITION BY mmsi1, track1, mmsi2, track2) AS avg_accel
#             FROM {1}
#             WHERE encounter = 'overtaking'
#         """.format(self.table, self.input)
#         self.cur.execute(sql)
#         self.conn.commit()
#
#     def separate_colregs(self):
#         '''Create table of colregs compliant interactions.'''
#         print('Selecting colreg compliant interactions...')
#         sql_colregs = """
#             CREATE TABLE {0} AS
#             SELECT *
#             FROM {1}
#             WHERE avg_cog_cos >= 0.999
#             AND avg_accel <= abs(10)
#         """.format(self.colregs, self.table)
#         self.cur.execute(sql_colregs)
#         self.conn.commit()
#
#         print('Selecting colreg compliant interactions...')
#         sql_others = """
#             CREATE TABLE {0} AS
#             SELECT *
#             FROM {1}
#             WHERE avg_cog_cos < 0.999
#             AND avg_accel > abs(10)
#         """.format(self.others, self.table)
#         self.cur.execute(sql_others)
#         self.conn.commit()
#
#

#
#
# # -------------def build_grid(self):
#     # '''Create grid table.'''
#     # print('Constructing grid table...')
#     # self.grid_df = dataframes.Sector_Dataframe(
#     #     self.lonMin,
#     #     self.lonMax,
#     #     self.latMin,
#     #     self.latMax,
#     #     self.stepSize
#     # ).generate_df()
#     # self.grid_csv = join(self.root, 'grid_table.csv')
#     # self.grid_df.to_csv(self.grid_csv, index=True, header=False)
#     # self.table_grid.drop_table()
#     #
#     # self.table_grid.create_table()
#     # self.table_grid.copy_data(self.grid_csv)
#     # self.table_grid.add_points()
#     # self.table_grid.make_bounding_box()----------------------------------------------------------------
#
# # class Near_Table(Postgres_Table):
# #
# #     def __init__(self, conn, table, input_table):
# #         '''Connect to default database.'''
# #         super(Near_Table, self).__init__(conn, table)
# #         self.cur = self.conn.cursor()
# #         self.input = input_table
# #
# #         self.columns =  [
# #             'own_mmsi',
# #             'own_sectorid',
# #             'own_sog',
# #             'own_heading',
# #             'own_rot',
# #             'own_length',
# #             'own_vesseltype',
# #             'own_tss',
# #             'target_mmsi',
# #             'target_sectorid',
# #             'target_sog',
# #             'target_heading',
# #             'target_rot',
# #             'target_vesseltype',
# #             'target_tss',
# #             'azimuth_deg',
# #             'point_distance',
# #             'bearing'
# #         ]
# #         self.columnString =  ', '.join(self.columns[:-1])
# #
# #     def near_points(self):
# #         '''Make pairs of points that happen in same sector and time interval.'''
# #         print('Joining nais_points with itself to make near table...')
# #         sql = """
# #             CREATE TABLE {0} AS
# #             SELECT
# #                 n1.mmsi AS own_mmsi,
# #                 n1.sectorid AS own_sectorid,
# #                 n1.trackid AS own_trackid,
# #                 n1.sog AS own_sog,
# #                 n1.cog AS own_cog,
# #                 n1.heading AS own_heading,
# #                 n1.rot AS own_rot,
# #                 n1.vesseltype AS own_vesseltype,
# #                 n1.length AS own_length,
# #                 n1.geom AS own_geom,
# #                 n1.in_tss AS own_tss,
# #                 n2.mmsi AS target_mmsi,
# #                 n2.sectorid AS target_sectorid,
# #                 n2.trackid AS target_trackid,
# #                 n2.sog AS target_sog,
# #                 n2.cog AS target_cog,
# #                 n2.heading AS target_heading,
# #                 n2.rot AS target_rot,
# #                 n2.vesseltype AS target_vesseltype,
# #                 n2.length AS target_length,
# #                 n2.geom AS target_geom,
# #                 n2.in_tss AS target_tss,
# #                 ST_Distance(ST_Transform(n1.geom, 7260), ST_Transform(n2.geom, 7260)) AS point_distance,
# #                 DEGREES(ST_Azimuth(n1.geom, n2.geom)) AS azimuth_deg
# #             FROM {1} n1 INNER JOIN {2} n2
# #             ON n1.sectorid = n2.sectorid
# #             WHERE n1.basedatetime = n2.basedatetime
# #             AND n1.mmsi != n2.mmsi
# #         """.format(self.table, self.input, self.input)
# #         self.cur.execute(sql)
# #         self.conn.commit()
# #
# #     def near_points_dataframe(self, max_distance):
#         '''Return dataframe of near points within max distance.'''
#         cond = 'point_distance <= {0}'.format(max_distance)
#         return self.table_dataframe(self.columnString, cond)
#
#     def near_table(self, max_distance):
#         '''Create near table in pandas using max_distance = 1nm.'''
#         # 1nm = 1852 meters
#         df = self.near_points_dataframe(max_distance)
#         df['bearing'] = (df['azimuth_deg'] - df['own_heading']) % 360
#         return df[self.columns].copy()
#
#     def near_plot(self, max_distance, display_points):
#         '''Plot the near points all in reference to own ship.'''
#         self.df_near = self.near_table(max_distance).head(display_points)
#         self.df_near = self.df_near[
#             (self.df_near['own_sog'] >10) & (self.df_near['target_sog'] > 10)].copy()
#         theta = np.array(self.df_near['bearing'])
#         r = np.array(self.df_near['point_distance'])
#         tss = np.array(self.df_near['bearing'])
#         # colors = theta.apply(np.radians)
#         colors = tss
#         fig = plt.figure()
#         ax = fig.add_subplot(111, projection='polar')
#         c = ax.scatter(theta, r, c=colors, cmap='hsv', alpha=0.75, s=1)
#         # plt.legend(loc='upper left')
#         plt.show()
#


# class Grid_Table(Postgres_Table):
#
#     def __init__(self, conn, table):
#         '''Connect to default database.'''
#         super(Grid_Table, self).__init__(conn, table)
#         self.cur = self.conn.cursor()
#
#         self.columns = """
#             SectorID char(5) PRIMARY KEY,
#             MinLon float8 NOT NULL,
#             MinLat float8 NOT NULL,
#             MaxLon float8 NOT NULL,
#             MaxLat float8 NOT NULL
#         """
#
#     def add_points(self):
#         '''Add Point geometry to the database.'''
#         self.add_column('leftBottom', datatype='POINT', geometry=True)
#         self.add_column('leftTop', datatype='POINT', geometry=True)
#         self.add_column('rightTop', datatype='POINT', geometry=True)
#         self.add_column('rightBottom', datatype='POINT', geometry=True)
#
#         print('Adding PostGIS POINTs to {0}...'.format(self.table))
#         self.add_point('leftBottom', 'minlon', 'minlat')
#         self.add_point('leftTop', 'minlon', 'maxlat')
#         self.add_point('rightTop', 'maxlon', 'maxlat')
#         self.add_point('rightBottom', 'maxlon', 'minlat')
#
#     def make_bounding_box(self):
#         '''Add polygon column in order to do spatial analysis.'''
#         print('Adding PostGIS POLYGON to {0}...'.format(self.table))
#         self.add_column('boundingbox', datatype='Polygon', geometry=True)
#         sql = """
#             UPDATE {0}
#             SET {1} = ST_SetSRID(ST_MakePolygon(
#                 ST_MakeLine(array[{2}, {3}, {4}, {5}, {6}])
#             ), 4326)
#         """.format(
#             self.table,
#             'boundingbox',
#             'leftBottom',
#             'leftTop',
#             'rightTop',
#             'rightBottom',
#             'leftBottom'
#             )
#         self.cur.execute(sql)
#         self.conn.commit()

    # def track_changes(self):
    #     '''Reorganize data into give way and stand on.'''
    #     print('Adding course info to {0}'.format(self.table))
    #     cog_gw = 'give_way_cog_cos'
    #     cog_so = 'stand_on_cog_cos'
    #     accel_gw = 'give_way_accel'
    #     accel_so = 'stand_on_accel'
    #     type_gw = 'give_way_type'
    #     type_so = 'stand_on_type'
    #
    #     self.add_column(cog_gw, datatype='float(4)')
    #     self.add_column(cog_so, datatype='float(4)')
    #     self.add_column(accel_gw, datatype='float(4)')
    #     self.add_column(accel_so, datatype='float(4)')
    #     self.add_column(type_gw, datatype='varchar')
    #     self.add_column(type_so, datatype='varchar')
    #
    #     sql = """
    #         UPDATE {table}
    #         SET
    #             {cog_gw} = CASE
    #                 WHEN ship1_give_way = 1 THEN cog_cos1 ELSE cog_cos2 END,
    #             {cog_so} = CASE
    #                 WHEN ship1_give_way = 1 THEN cog_cos2 ELSE cog_cos1 END,
    #             {accel_gw} = CASE
    #                 WHEN ship1_give_way = 1 THEN accel1 ELSE accel2 END,
    #             {accel_so} = CASE
    #                 WHEN ship1_give_way = 1 THEN accel2 ELSE accel1 END,
    #             {type_gw} = CASE
    #                 WHEN ship1_give_way = 1 THEN type1 ELSE type2 END,
    #             {type_so} = CASE
    #                 WHEN ship1_give_way = 1 THEN type2 ELSE type1 END
    #     """.format(
    #         table=self.table,
    #         cog_gw=cog_gw,
    #         cog_so=cog_so,
    #         accel_gw=accel_gw,
    #         accel_so=accel_so,
    #         type_gw=type_gw,
    #         type_so=type_so
    #     )
    #     self.cur.execute(sql)
    #     self.conn.commit()
    #
