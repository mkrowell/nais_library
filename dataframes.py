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
import matplotlib.pyplot as plt
import numpy as np
from os.path import basename, dirname, join
import pandas as pd
import seaborn as sns
sns.set_palette("Set1")
sns.set_context("paper", font_scale=3)
sns.axes_style("darkgrid")

from . import check_length, print_reduction, time_all, concat_df


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

def angle_difference(angle1, angle2):
    '''Return the signed difference between two angles.'''
    angle1 = np.radians(angle1)
    angle2 = np.radians(angle2)
    y = np.sin(angle1 - angle2)
    x = np.cos(angle1 - angle2)
    return np.arctan2(y, x)


# ------------------------------------------------------------------------------
# DATAFRAMES
# ------------------------------------------------------------------------------
@time_all
class NAIS_Dataframe(object):

    '''
    Clean raw NAIS csv file by removing invalid and unneccessary data. Add
    additional derived columns to help in later analysis.
    '''

    def __init__(self, csvFiles, lonMin, lonMax, latMin, latMax):
        '''Process nais dataframe.'''
        self.csvs = csvFiles
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
            'Width'
        ]
        self.headers_required = [
            'MMSI',
            'BaseDateTime',
            'LAT',
            'LON',
            'SOG',
            'COG',
            'VesselType'
        ]
        self.df = concat_df(pd.read_csv, self.csvs, usecols=self.headers)
        self.df['BaseDateTime'] = pd.to_datetime(self.df['BaseDateTime'])
        self.df.sort_values(['MMSI','BaseDateTime'], inplace=True)

        self.df['Status'].replace(np.nan, "undefined", inplace=True)
        self.df['Status'] = self.df['Status'].astype('category')

        # Normalize COG
        self.normalize_angle('COG', 0, 360)
        self.df.drop(columns='COG', inplace=True)
        self.df.rename(columns={'COG_Normalized':'COG'}, inplace=True)


    # PROPERTIES ---------------------------------------------------------------
    @property
    def grouped_mmsi(self):
        '''Return sorted dataframe grouped by MMSI.'''
        return self.df.sort_values(['MMSI', 'BaseDateTime']).groupby('MMSI')

    @property
    def grouped_time(self):
        '''Return sorted dataframe grouped by MMSI and Time Track.'''
        return self.df.sort_values(
            ['MMSI', 'Track', 'BaseDateTime']
        ).groupby(
            ['MMSI', 'Track']
        )


    # MAIN FUNCTIONS -----------------------------------------------------------
    @print_reduction
    def clean(self, sensitivity=0.15, maxJump=4):
        '''Clean bad or unneccessary data from dataframe.'''
        self.drop_spatial()
        self.drop_null()
        self.drop_bad_mmsi()
        self.drop_duplicate_keys()
        self.drop_bad_speed()
        self.drop_bad_heading()
        self.drop_vessel_types()

        # add check columns
        self.point_cog()
        self.step_time()
        self.step_distance()
        self.expected_distance()

        # drop suspicious data
        self.drop_bad_distance(sensitivity)
        self.mark_time_jump(maxJump)
        self.drop_jump_string()

    def split_mmsi_stop(self, maxTime=2):
        '''Split tracks into stopped and moving segments.'''
        self.mark_stop(maxTime)
        self.drop_stops()

        self.step_time()
        self.mark_time_jump(maxJump=4)
        self.mark_segment()
        self.drop_sparse_track()

    def add_evasive_data(self):
        '''Add changes in speed and course.'''
        self.step_acceleration()
        self.step_cog()
        self.step_heading()

    def save_output(self):
        '''Save output to be read into PostgreSQL.'''
        self.validate_types()
        self.normalize_time()
        self.reorder_output()


    # CLEAN DATA ---------------------------------------------------------------
    @print_reduction
    def drop_spatial(self):
        '''Limit to area of interest's bounding box.'''
        self.df = self.df[self.df['LON'].between(self.lonMin, self.lonMax)].copy()
        self.df = self.df[self.df['LAT'].between(self.latMin, self.latMax)].copy()

    @print_reduction
    def drop_null(self):
        '''Drop rows with nulls in the required columns.'''
        for col in self.headers_required:
            self.df[col].replace("", np.nan, inplace=True)
        self.df.dropna(how='any', subset=self.headers_required, inplace=True)

    @print_reduction
    def drop_bad_mmsi(self):
        '''MMSI numbers should be 9 digits and between a given range.'''
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
    def drop_vessel_types(self):
        '''Map codes to categories.'''
        types = {
            31: 'tug',
            32: 'tug',
            52: 'tug',
            60: 'passenger',
            61: 'passenger',
            62: 'passenger',
            63: 'passenger',
            64: 'passenger',
            65: 'passenger',
            66: 'passenger',
            67: 'passenger',
            68: 'passenger',
            69: 'passenger',
            70: 'cargo',
            71: 'cargo',
            72: 'cargo',
            73: 'cargo',
            74: 'cargo',
            75: 'cargo',
            76: 'cargo',
            77: 'cargo',
            78: 'cargo',
            79: 'cargo',
            80: 'tanker',
            81: 'tanker',
            82: 'tanker',
            83: 'tanker',
            84: 'tanker',
            85: 'tanker',
            86: 'tanker',
            87: 'tanker',
            88: 'tanker',
            89: 'tanker',
            1003: 'cargo',
            1004: 'cargo',
            1012: 'passenger',
            1014: 'passenger',
            1016: 'cargo',
            1017: 'tanker',
            1023: 'tug',
            1024: 'tanker',
            1025: 'tug'
        }
        codes = list(types.keys())
        self.df = self.df[self.df['VesselType'].isin(codes)].copy()
        self.df['VesselType'] = self.df['VesselType'].map(types)
        self.df['VesselType'] = self.df['VesselType'].astype('category')

    @print_reduction
    def drop_bad_speed(self):
        '''SOG should be positive.'''
        bad_mmsi = self.df[self.df['SOG'] < 0]['MMSI'].unique().tolist()
        self.df = self.df[~self.df['MMSI'].isin(bad_mmsi)].copy()

    @print_reduction
    def drop_bad_heading(self):
        '''Drop undefined heading.'''
        self.df['Heading'].replace(511, np.nan, inplace=True)
        self.df.dropna(how='any', subset=['Heading'], inplace=True)

    def point_cog(self):
        '''Calculate the course between two position points.'''
        def course(df):
            df.reset_index(inplace=True)
            df['Point_COG'] = azimuth(
                df['LAT'].shift(),
                df['LON'].shift(),
                df.loc[1:,'LAT'],
                df.loc[1:,'LON']
            )
            return df.set_index('index')
        self.df = self.grouped_mmsi.apply(course)
        self.df['Point_COG'].fillna(method='bfill', inplace=True)
        self.normalize_angle('Point_COG', 0, 360)
        self.df.drop(columns=['Point_COG'], inplace=True)
        self.df.rename(columns={'Point_COG_Normalized': 'Point_COG'}, inplace=True)

    @print_reduction
    def drop_bad_cog(self):
        '''Remove bad COG recordings.'''
        self.df['Error_COG'] = abs(self.df['COG'] - self.df['Point_COG'])
        self.df = self.df[self.df['Error_COG']<5].copy()

    @print_reduction
    def drop_sparse_mmsi(self):
        '''Remove MMSIs that have less than 5 data points.'''
        self.df = self.df.groupby(['MMSI']).filter(lambda g: len(g) >= 5)

    def step_time(self):
        '''Return time between timestamps.'''
        self.df['Step_Time'] = self.grouped_mmsi['BaseDateTime'].diff()
        self.df['Step_Time'] = self.df['Step_Time'].astype('timedelta64[s]')
        self.df['Step_Time'].fillna(method='bfill', inplace=True)

    def step_distance(self):
        '''Return distance between timestamps.'''
        def distance(df):
            df.reset_index(inplace=True)
            df['Step_Distance'] = haversine(
                df['LAT'].shift(),
                df['LON'].shift(),
                df.loc[1:,'LAT'],
                df.loc[1:,'LON'])
            return df.set_index('index')
        self.df = self.grouped_mmsi.apply(distance)
        self.df['Step_Distance'].fillna(method='bfill', inplace=True)

    def expected_distance(self):
        '''Calculate expected distance given speed and time.'''
        self.df['Expected_Distance'] = self.df['SOG']*self.df['Step_Time']/3600

    @print_reduction
    def drop_bad_distance(self, sensitivity):
        '''Drop distances outside expected values.'''
        max = (1 + sensitivity)*self.df['Expected_Distance']
        cond_distance = self.df['Step_Distance'] > max
        # At infrequent time intervals, speed is ~0, distance is less reliable
        cond_time = self.df['Step_Time'] < 120
        self.df['Drop'] = np.where((cond_distance) & (cond_time), 1, 0)

        self.df = self.df[self.df['Drop']==0].copy()
        self.df.drop(columns=['Drop'], inplace=True)

    def mark_time_jump(self, maxJump):
        '''Mark points with large time jump.'''
        def mark(df):
            df.reset_index(inplace=True)
            df['Time_Jump'] = np.where(df['Step_Time'] > maxJump*60, 1, 0)
            return df.set_index('index')
        self.df = self.grouped_mmsi.apply(mark)

    @print_reduction
    def drop_jump_string(self):
        '''Drop consecutive time jumps.'''
        self.df['Jump_String'] = np.where(
            (self.df['Time_Jump'] == 1) & (self.df['Time_Jump'].shift() == 1),
            1,
            0
        )
        self.df = self.df[self.df['Jump_String'] == 0].copy()
        self.df.drop(columns=['Jump_String'], inplace=True)

        for col in ['Step_Time', 'Step_Distance']:
            self.df[col] = np.where(self.df['Time_Jump'] == 1,
                self.df[col].shift(),
                self.df[col]
            )


    # STOPS --------------------------------------------------------------------
    def mark_stop(self, maxTime=2):
        '''Assign status 'stop' to a point if it satisfies criteria'''
        # Transmission frequency over 2 minutes
        cond_time = (self.df['Step_Time'] > maxTime*60)
        cond_speed = (self.df['SOG'] == 0)
        self.df['Stop'] = np.where(cond_time | cond_speed, 1, 0)

        # Status is stopped
        status_stop = ['not under command', 'at anchor', 'moored', 'aground']
        self.df['Stop'] = np.where(
            self.df['Status'].isin(status_stop),
            1,
            self.df['Stop']
        )

    @print_reduction
    def drop_stops(self):
        '''Drop stops.'''
        self.df = self.df[self.df['Stop']==0].copy()

    def mark_segment(self):
        '''Assign an id to points .'''
        self.df['Track'] = self.grouped_mmsi['Time_Jump'].cumsum()
        self.df['Track'] = self.df['Track'].astype(int)
        self.df.drop(columns=['Time_Jump'], inplace=True)
        self.df.sort_values(['MMSI', 'Track', 'BaseDateTime'], inplace=True)

    @print_reduction
    def drop_sparse_track(self):
        '''Remove MMSIs that have less than 30 data points.'''
        grouped =  self.df.groupby(['MMSI', 'Track'])
        self.df = grouped.filter(lambda g: g['SOG'].mean() >= 2)
        self.df = grouped.filter(lambda g: len(g) >= 30)


    # STEP CALCULATIONS --------------------------------------------------------
    def step_acceleration(self):
        '''Add acceleration field.'''
        self.df['DS'] = self.grouped_time['SOG'].diff()
        self.df['Step_Acceleration'] = 3600*self.df['DS'].divide(
            self.df['Step_Time'], fill_value=0)
        self.df.drop(columns=['DS'], inplace=True)
        self.df['Step_Acceleration'].fillna(method='bfill', inplace=True)

    def step_cog(self):
        '''Calculate change in course.'''
        def delta_course(df):
            df.reset_index(inplace=True)
            df['Step_COG_Radians'] = angle_difference(
                df['Point_COG'].shift(),
                df.loc[1:,'Point_COG']
            )
            return df.set_index('index')
        self.df = self.grouped_time.apply(delta_course)
        self.df['Step_COG_Radians'].fillna(method='bfill', inplace=True)
        self.df['COG_Cosine'] = np.cos(self.df['Step_COG_Radians'])
        self.df['Step_COG_Degrees'] = np.degrees(self.df['Step_COG_Radians'])


    # PREP FOR POSTGRES --------------------------------------------------------
    def validate_types(self):
        '''Cast to correct data types.'''
        cols_float = ['Length', 'Width']
        for col in cols_float:
            self.df[col].fillna(-1, inplace=True)
            self.df[col] = self.df[col].astype(float)

    @print_reduction
    def normalize_time(self):
        '''Round time to nearest minute.'''
        self.df['BaseDateTime'] = self.df['BaseDateTime'].dt.round('1min')
        self.df.drop_duplicates(
            subset=['MMSI', 'BaseDateTime'],
            keep='first',
            inplace=True
        )

    def reorder_output(self):
        '''Save processed df to csv file.'''
        order = [
            'MMSI',
            'BaseDateTime',
            'Track',
            'Step_COG_Degrees',
            'Step_COG_Radians',
            'COG_Cosine',
            'Step_Acceleration',
            'LAT',
            'LON',
            'SOG',
            'COG',
            'Point_COG',
            'Heading',
            'VesselName',
            'VesselType',
            'Status',
            'Length',
            'Width'
        ]
        output = self.df[order].copy()
        output.to_csv(
            join(dirname(self.csvs[0]), 'AIS_All.csv'),
            index=False,
            header=False
        )


    # HELPER FUNCTIONS ---------------------------------------------------------
    def normalize_angle(self, column, start, end):
        '''Normalized an angle to be within the start and end.'''
        width = end - start
        offset = self.df[column] - start
        name = '{0}_Normalized'.format(column)
        self.df[name] = offset - np.floor(offset/width)*width + start


