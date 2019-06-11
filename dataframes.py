#!/usr/bin/env python
'''
.. module::
    :language: Python Version 3.6.8
    :platform: Windows 10
    :synopsis: process raw NAIS broadcast data points

.. moduleauthor:: Maura Rowell <mkrowell@uw.edu>
'''


# ------------------------------------------------------------------------------
# IMPORTS
# ------------------------------------------------------------------------------
from geographiclib.geodesic import Geodesic
import geopy.distance
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import check_length, print_reduction, time_all


# ------------------------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------------------------
METERS_IN_NM = 1852
EARTH_RADIUS_KM = 6371


# ------------------------------------------------------------------------------
# FUNCTIONS
# ------------------------------------------------------------------------------
def haversine(lat1, lon1, lat2, lon2):
    '''Return the haversine distance between two points.'''
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    a = np.sin((lat2-lat1)/2.0)**2 + \
        np.cos(lat1) * np.cos(lat2) * np.sin((lon2-lon1)/2.0)**2
    return (EARTH_RADIUS_KM * 2 * np.arcsin(np.sqrt(a)))/(METERS_IN_NM/1000)

def azimuth(lat1, lon1, lat2, lon2):
    '''Return the bearing between two points.'''
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    y = np.sin(lon2 - lon1) * np.cos(lat2)
    x = np.cos(lat1)*np.sin(lat2)-np.sin(lat1)*np.cos(lat2)*np.cos(lon2 - lon1)
    return np.degrees(np.arctan2(y, x))


# ------------------------------------------------------------------------------
# DATAFRAMES
# ------------------------------------------------------------------------------
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


    def grid(self, array, i):
        '''Construct grid.'''
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

@time_all
class NAIS_Dataframe(object):

    '''
    Clean raw NAIS csv file by removing invalid and unneccessary data. Add
    additional derived columns to help in later analysis.
    '''

    def __init__(self, csvFile, lonMin, lonMax, latMin, latMax):
        '''Process nais dataframe.'''
        self.csv = csvFile
        self.lonMin = lonMin
        self.lonMax = lonMax
        self.latMin = latMin
        self.latMax = latMax

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
            'LAT',
            'LON',
            'SOG',
            'COG',
            'Heading',
            'VesselType'
        ]
        # self.sortBy = ['MMSI', 'BaseDateTime']
        # self.sortTimeBy = ['MMSI', 'Time_Track', 'BaseDateTime']

        self.df = pd.read_csv(self.csv, usecols=self.headers)
        self.df['BaseDateTime'] = pd.to_datetime(self.df['BaseDateTime'])

    # PROPERTIES ---------------------------------------------------------------
    @property
    def grouped_mmsi(self):
        '''Return sorted dataframe grouped by MMSI.'''
        return self.df.sort_values(['MMSI', 'BaseDateTime']).groupby('MMSI')

    @property
    def grouped_time(self):
        '''Return sorted dataframe grouped by MMSI and Time Track.'''
        groupBy = ['MMSI', 'Time_Track']
        return self.df.sort_values(
            ['MMSI', 'Track_AIS', 'BaseDateTime']
        ).groupby(
            ['MMSI', 'Track_AIS']
        )

    # HELPER FUNCTIONS ---------------------------------------------------------
    # def normalize_angle(self, column, start, end):
    #     '''Normalized an angle to be within the start and end.'''
    #     width = end - start
    #     offset = self.df[column] - start
    #     name = '{0}_Normalized'.format(column)
    #     self.df[name] = offset - np.floor(offset/width)*width + start

    def normalize_angle(self, column, start, end):
        '''Normalized an angle to be within the start and end.'''
        self.df['Difference'] = self.grouped_time[column].diff()
        width = end - start
        offset = self.df['Difference'] - start
        name = '{0}_Normalized'.format(column)
        self.df[name] = offset - np.floor(offset/width)*width + start
        self.df.drop(columns=['Difference'], inplace=True)


    # MAIN FUNCTIONS -----------------------------------------------------------
    @print_reduction
    def clean(self):
        '''Remove bad data from dataframe.'''
        # Handle keys
        self.drop_bad_mmsi()
        self.drop_duplicate_keys()

        # Drop out of scope or missing data
        self.drop_spatial()
        self.drop_null()

        # Drop invalid data
        self.drop_bad_speed()
        self.drop_bad_heading()

        # Drop sparse data
        self.drop_sparse_mmsi()

        # Types
        self.cast_columns()

    def split_mmsi_jump(self, maxTime=5, maxDistance=0.2):
        '''Split MMSI data over large time gaps.'''
        self.step_time()
        self.step_distance()
        self.mark_time_jump(maxTime)
        self.mark_distance_jump(maxDistance)
        self.mark_jump()
        self.drop_sparse_track()

    def split_mmsi_stop(self, maxTime=2, minDisplace=5, minAccel=10, minSpeed=2):
        '''Split MMSI data over stops.'''
        self.step_acceleration()
        self.step_displacement()
        self.step_rot()
        self.mark_stop(maxTime, minDisplace, minAccel, minSpeed)
        self.mark_segment()


    # CLEAN DATA ---------------------------------------------------------------
    @print_reduction
    def drop_bad_mmsi(self):
        '''MMSI numbers should be 9 digits.'''
        condLen = self.df['MMSI'].apply(lambda x: len(str(x)) == 9)
        self.df = self.df[condLen].copy()

        condRange = self.df['MMSI'].between(200999999, 776000000)
        self.df = self.df[condRange].copy()

    @print_reduction
    def drop_duplicate_keys(self):
        '''MMSI, BaseDateTime are primary key, should be unique.'''
        self.df.drop_duplicates(
            subset=['MMSI', 'BaseDateTime'],
            keep=False,
            inplace=True
        )

    @print_reduction
    def drop_spatial(self):
        '''Limit to area of interest's bounding box.'''
        self.df = self.df[self.df['LON'].between(
            self.lonMin, self.lonMax, inclusive=False
        )].copy()
        self.df = self.df[self.df['LAT'].between(
            self.latMin, self.latMax, inclusive=False
        )].copy()

    @print_reduction
    def drop_bad_speed(self):
        '''SOG should be positive.'''
        self.df = self.df[self.df['SOG'] >= 0].copy()

    @print_reduction
    def drop_bad_heading(self):
        '''Headings must be between 0 and 360.'''
        self.df['Heading'] = np.where(
            (self.df['Heading'] > 360) & (self.df['Heading'] > 0),
            np.nan,
            self.df['Heading']
        )
        self.df.dropna(subset=['Heading'], inplace=True)

    @print_reduction
    def drop_null(self):
        '''''Drop rows with nulls in the required columns.'''
        for col in self.headers_required:
            self.df[col].replace("", np.nan, inplace=True)
        self.df.dropna(how='any', subset=self.headers_required, inplace=True)

    @print_reduction
    def drop_sparse_mmsi(self):
        '''Remove MMSIs that have less than 50 data points.'''
        self.df = self.df.groupby(['MMSI']).filter(lambda g: len(g) >= 50)

    def cast_columns(self):
        for col in ['VesselType', 'Cargo', 'Status']:
            self.df[col].astype('category')


    # AIS JUMPS ----------------------------------------------------------------
    @check_length
    def step_time(self):
        '''Return time between timestamps.'''
        self.df['Step_Time'] = self.grouped_mmsi['BaseDateTime'].diff()
        self.df['Step_Time'] = self.df['Step_Time'].astype('timedelta64[s]')

    @check_length
    def step_distance(self):
        '''Return distance between timestamps.'''
        def distance(df):
            df.reset_index(inplace=True)
            df['Step_Distance'] =  haversine(
                df['LAT'].shift(),
                df['LON'].shift(),
                df.loc[1:]['LAT'],
                df.loc[1:]['LON'])
            return df.set_index('index')
        self.df = self.grouped_mmsi.apply(distance)

    def mark_time_jump(self, maxJump):
        '''Mark points with large time jump.'''
        def mark(df):
            df.reset_index(inplace=True)
            df['Break_Time'] = np.where(
                df['Step_Time'] > maxJump*60,
                1,
                0)
            return df.set_index('index')
        self.df = self.grouped_mmsi.apply(mark)

    def mark_distance_jump(self, maxJump):
        '''Mark points with large distance jump.'''
        def mark(df):
            df.reset_index(inplace=True)
            df['Break_Distance'] = np.where(
                df['Step_Distance'] > maxJump,
                1,
                0)
            return df.set_index('index')
        self.df = self.grouped_mmsi.apply(mark)

    def mark_jump(self):
        '''Mark points that have either a time or distance jump.'''
        self.df['Break_AIS'] = np.where(
            (self.df['Break_Time']==1) | (self.df['Break_Distance']==1), 1, 0)
        self.df['Track_AIS'] = self.grouped_mmsi['Break_AIS'].cumsum()

        # Reset distance and time so I don't catch again
        self.df['Step_Time'] = np.where(
            self.df['Break_AIS'] == 1,
            np.nan,
            self.df['Step_Time'])
        self.df['Step_Distance'] = np.where(
            self.df['Break_AIS'] == 1,
            np.nan,
            self.df['Step_Distance'])

        self.df.drop(columns=['Break_Time','Break_Distance'], inplace=True)

    @print_reduction
    def drop_sparse_track(self):
        '''Drop sparse tracks.'''
        self.df = self.grouped_time.filter(lambda g: len(g)>=50)


    # STOP BREAKS --------------------------------------------------------------
    def step_acceleration(self):
        '''Add acceleration field.'''
        self.df['DS'] = self.grouped_time['SOG'].diff().fillna(0)
        self.df['Step_Acceleration'] = 3600*self.df['DS'].divide(
            self.df['Step_Time'], fill_value=np.inf
        )
        self.df.drop(columns=['DS'], inplace=True)

    @check_length
    def step_displacement(self):
        '''Calculate the relative net displacement between rows.'''
        def distance(df):
            df.reset_index(inplace=True)
            df['Net_Displacement'] =  haversine(
                df.loc[1].at['LAT'],
                df.loc[1].at['LON'],
                df.loc[1:]['LAT'],
                df.loc[1:]['LON']
            )
            return df.set_index('index')
        self.df = self.grouped_time.apply(distance)

        self.df['Step_Displacement'] = self.df['Net_Displacement'].pct_change()
        self.df['Step_Displacement'].replace(np.inf, np.nan, inplace=True)
        self.df['Step_Displacement'].fillna(0, inplace=True)
        self.df.drop(columns=['Net_Displacement'], inplace=True)

    def step_rot(self):
        '''Add ROT field.'''
        self.normalize_angle('Heading', -180, 180)
        self.df['Heading_Normalized'].fillna(0, inplace=True)
        self.df['Step_ROT'] = 60*self.df['Heading_Normalized'].divide(
            self.df['Step_Time'], fill_value=np.inf
        )
        self.df.drop(columns=['Heading_Normalized'], inplace=True)

    def mark_stop(self, maxTime, minDisplace, minAccel, minSpeed):
        '''Assign status 'stop' to a point if it satisfies criteria'''
        # How is NA handled here?
        cTime = (self.df['Step_Time'] > maxTime*60)
        cAccel = (abs(self.df['Step_Acceleration']) < minAccel)
        cSOG = (self.df['SOG'] < minSpeed)
        cDisplace = (abs(self.df['Step_Displacement']) < minDisplace/100)
        cond = cTime & cSOG & cDisplace & cAccel
        self.df['Stop'] = np.where(cond, 1, 0)

    def mark_segment(self):
        '''Assign an id to points .'''
        self.df['Stop_Change'] = abs(self.df['Stop'].diff())
        self.df['Break_Stop'] = self.grouped_mmsi['Stop_Change'].cumsum()
        self.df['Track_MMSI'] = self.df['Track_AIS'] + self.df['Break_Stop']


    # PREP FOR POSTGRES --------------------------------------------------------
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
        cols_int = ['VesselType', 'Cargo', 'Segment']
        for col in cols_int:
            self.df[col] = self.df[col].astype(int)

    def reorder_output(self):
        '''Save processed df to csv file.'''
        order = [
            'MMSI',
            'BaseDateTime',
            'Track_MMSI',
            'LAT',
            'LON',
            'SOG',
            'COG',
            'Heading',
            'Step_ROT',
            'Step_Acceleration',
            'Stop',
            'Step_Displacement',
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



    def plots(self):
        '''Plot time lag.'''
        plt.style.use(['ggplot', 'presentation'])
        fig = plt.figure()

        ax2 = plt.subplot(111)
        ax2.title.set_text('Step Time v Step Distance')
        ax2.set_xlabel('Step_Time')
        ax2.set_ylabel('Step_Distance')
        plt.scatter(
            self.df['Step_Time'].astype('timedelta64[s]'),
            self.df['Step_Distance'])

        plt.show()


    # @check_length
    # def step_bearing(self):
    #     '''Return bearing required to get from first to second point.'''
    #     def bearing(df):
    #         df.reset_index(inplace=True)
    #         df['Step_Bearing'] = azimuth(
    #             df['LAT'].shift(),
    #             df['LON'].shift(),
    #             df.iloc[1:]['LAT'],
    #             df.iloc[1:]['LON']
    #         )
    #         return df.set_index('index')
    #     self.df = self.grouped_mmsi.apply(bearing)
    #     self.normalize_angle('Step_Bearing', 0, 360)
