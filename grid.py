#!/usr/bin/env python
'''
.. module::
    :language: Python Version 3.6.8
    :platform: Windows 10
    :synopsis: select space-time grid rows within database

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

from . import database


# ------------------------------------------------------------------------------
# PARAMETERS
# ------------------------------------------------------------------------------
test_min_lon = -123.50222222
test_max_lon = -123.29916667
test_min_lat = 48.29916667
test_max_lat = 48.19750000
min_date = datetime.datetime(2017, 1, 1, 0, 0, 0)
max_date = datetime.datetime(2017, 12, 31, 23, 59, 59)

# ------------------------------------------------------------------------------
# MAIN CLASS
# ------------------------------------------------------------------------------
class NAIS_Grid(object):

    def __init__(self, min_lon, max_lon, min_lat, max_lat, min_date, max_date):

        self.min_lon = min_lon
        self.max_lon = max_lon
        self.min_lat = min_lat
        self.max_lat = max_lat
        self.min_date = min_date
        self.max_date = max_date

        # Initialize raw table
        self.db = database.NAIS_Table(password, 'nais_points')

        # Select this region from nais_points
        self.db.cur.execute("DROP TABLE IF EXISTS {0}".format('nais_tracks_1'))
        self.db.cur.execute("""
            CREATE TABLE nais_tracks_1 AS
            SELECT mmsi, trackid, ST_MakeLine(geom ORDER BY basedatetime) AS track
            FROM nais_points
            WHERE lon BETWEEN {0} AND {1}
            AND lat BETWEEN {2} AND {3}
            AND basedatetime BETWEEN '{4}' AND '{5}'
            GROUP BY mmsi, trackid
            """.format(
                self.min_lon,
                self.max_lon,
                self.min_lat,
                self.max_lat,
                self.min_date.strftime('%Y-%m-%d %H:%M:%S'),
                self.max_date.strftime('%Y-%m-%d %H:%M:%S')
            )
        )
        self.db.conn.commit()
