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
            'MMSI',
            'BaseDateTime',
            'LAT',
            'LON',
            'SOG',
            'COG',
            'VesselType'
        ]

        self.df = pd.read_csv(self.csv, usecols=self.headers)
        self.df['BaseDateTime'] = pd.to_datetime(self.df['BaseDateTime'])
        self.df.sort_values(['MMSI','BaseDateTime'], inplace=True)


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
    def clean(self):
        '''Clean bad or unneccessary data from dataframe.'''
        # Handle keys
        self.drop_bad_mmsi()
        self.drop_duplicate_keys()

        # Drop out of scope or missing data
        self.drop_spatial()
        self.drop_null()
        self.drop_vessel_types()

        # Drop invalid data
        self.drop_bad_speed()

        # Drop sparse data
        self.drop_sparse_mmsi()

        # Types
        self.normalize_cog()
        self.cast_columns()

    def split_mmsi_jump(self, maxJump=4, sensitivty=0.35):
        '''Split MMSI data over large time and distance gaps.'''
        self.step_time()
        self.step_distance()
        self.expected_distance()
        self.dump_bad_distance(sensitivty)
        self.mark_time_jump(maxJump)
        self.dump_jump_string()

    def split_mmsi_stop(self, maxTime=2):
        self.mark_stop(maxTime)
        self.mark_segment()

    def add_evasive_data(self):
        '''Save output to be read into PostgreSQL.'''
        self.step_acceleration()
        self.step_cog()
        self.step_rot()

    def save_output(self):
        '''Save output to be read into PostgreSQL.'''
        self.validate_types()
        self.reorder_output()


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
    def drop_null(self):
        '''''Drop rows with nulls in the required columns.'''
        for col in self.headers_required:
            self.df[col].replace("", np.nan, inplace=True)
        self.df.dropna(how='any', subset=self.headers_required, inplace=True)

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
    def drop_vessel_types(self):
        '''Remove non-vessel and uscg'.'''
        bad_codes = [1008, 1009, 1018]
        self.df = self.df[~self.df['VesselType'].isin(bad_codes)].copy()

    @print_reduction
    def drop_bad_speed(self):
        '''SOG should be positive.'''
        bad_mmsi = self.df[self.df['SOG'] < 0]['MMSI'].unique().tolist()
        self.df = self.df[~self.df['MMSI'].isin(bad_mmsi)].copy()

    @print_reduction
    def drop_sparse_mmsi(self):
        '''Remove MMSIs that have less than 50 data points.'''
        self.df = self.df.groupby(['MMSI']).filter(lambda g: len(g) >= 2)

    def normalize_cog(self):
        '''Normalize COG to be between 0 and 360.'''
        self.normalize_angle('COG', 0, 360)

    def cast_columns(self):
        for col in ['VesselType', 'Cargo', 'Status']:
            self.df[col].astype('category')


    # AIS JUMPS ----------------------------------------------------------------
    def step_time(self):
        '''Return time between timestamps.'''
        self.df['Step_Time'] = self.grouped_mmsi['BaseDateTime'].diff()
        self.df['Step_Time'] = self.df['Step_Time'].astype('timedelta64[s]')
        self.df['Step_Time'].fillna(method='bfill', inplace=True)

    def step_distance(self):
        '''Return distance between timestamps.'''
        def distance(df):
            df.reset_index(inplace=True)
            df['Step_Distance'] =  haversine(
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
    def dump_bad_distance(self, sensitivity):
        '''Drop distances outside expected values.'''
        # Max distance is straight line at speed over time
        # Can be less if it is not going straight
        high = 1 + sensitivity
        cond_distance = self.df['Step_Distance'] > high*self.df['Expected_Distance']
        # At infrequent time intervals, speed is ~0 and distance is less reliable
        cond_time = self.df['Step_Time'] < 120
        self.df['Dump'] = np.where((cond_distance) & (cond_time),1,0)
        self.df = self.df[self.df['Dump']==0].copy()
        self.df.drop(columns=['Expected_Distance', 'Dump'], inplace=True)

    def mark_time_jump(self, maxJump):
        '''Mark points with large time jump.'''
        def mark(df):
            df.reset_index(inplace=True)
            df['Time_Jump'] = np.where(
                df['Step_Time'] > maxJump*60, 1, 0)
            return df.set_index('index')
        self.df = self.grouped_mmsi.apply(mark)

    @print_reduction
    def dump_jump_string(self):
        '''Drop consecutive time jumps.'''
        self.df['Jump_String'] = np.where(
            (self.df['Time_Jump'] == 1) & (self.df['Time_Jump'].shift() == 1),
            1,
            0
        )
        self.df = self.df[self.df['Jump_String'] == 0].copy()
        self.df['Step_Time'] = np.where(
            self.df['Time_Jump'] == 1, np.nan, self.df['Step_Time'])
        self.df['Step_Distance'] = np.where(
            self.df['Time_Jump'] == 1, np.nan, self.df['Step_Distance'])
        self.df.drop(columns=['Jump_String'], inplace=True)




    # STOPS --------------------------------------------------------------------
    def mark_stop(self, maxTime=2):
        '''Assign status 'stop' to a point if it satisfies criteria'''
        cond = (self.df['Step_Time'] > maxTime*60)
        self.df['Stop'] = np.where(cond, 1, 0)

    def mark_segment(self):
        '''Assign an id to points .'''
        self.df['Stop_Change'] = abs(self.grouped_mmsi['Stop'].diff()).fillna(0)
        self.df['Break'] = self.df['Time_Jump'] + self.df['Stop_Change']
        self.df['Track'] = self.grouped_mmsi['Break'].cumsum()
        self.df.drop(columns=['Time_Jump', 'Stop_Change', 'Break'], inplace=True)


    # CHANGE IN COURSE SPEED ---------------------------------------------------
    def step_acceleration(self):
        '''Add acceleration field.'''
        self.df['DS'] = self.grouped_time['SOG'].diff().fillna(0)
        self.df['Step_Acceleration'] = 3600*self.df['DS'].divide(
            self.df['Step_Time'], fill_value=np.inf
        )
        self.df.drop(columns=['DS'], inplace=True)

    @check_length
    def step_cog(self):
        '''Calculate the relative net displacement between rows.'''
        def course(df):
            df.reset_index(inplace=True)
            df['Step_COG'] = azimuth(
                df['LAT'].shift(),
                df['LON'].shift(),
                df.loc[1:,'LAT'],
                df.loc[1:,'LON']
            )
            return df.set_index('index')
        self.df = self.grouped_mmsi.apply(course)

        self.normalize_angle('Step_COG', 0, 360)
        self.df.drop(columns=['COG'], inplace=True)

    def step_rot(self):
        '''Add ROT field.'''
        self.normalize_angle_diff('Step_COG', -180, 180)
        self.df['Step_COG_Difference'].fillna(0, inplace=True)
        # self.df['Step_ROT'] = 60*self.df['Step_COG_Difference'].divide(
        #     self.df['Step_Time'], fill_value=np.inf
        # )
        self.df['Step_ROT'] = 60*self.df['Step_COG_Difference'].divide(
            self.df['Step_Time'])
        self.df.drop(columns=['step_COG', 'Step_COG_Difference'], inplace=True)


    # PREP FOR POSTGRES --------------------------------------------------------
    def validate_types(self):
        '''Cast to correct data types.'''
        # Replace empty elements with -1
        cols_null = list(set(self.df.columns.tolist()) - set(self.headers_required))
        for col in cols_null:
            self.df[col].fillna(-1, inplace=True)

        # Handle string columns
        self.df['Status'].replace(-1, "undefined", inplace=True)
        self.df['VesselName'].replace(-1, "undefined", inplace=True)

        # Handle int columns
        self.df['Heading'].replace(511, -1, inplace=True)
        cols_int = ['VesselType', 'Cargo', 'Track']
        for col in cols_int:
            self.df[col] = self.df[col].astype(int)

    def reorder_output(self):
        '''Save processed df to csv file.'''
        order = [
            'MMSI',
            'BaseDateTime',
            'Track',
            'Stop',
            'Step_ROT',
            'Step_Acceleration',
            'LAT',
            'LON',
            'SOG',
            'COG_Normalized',
            'Step_COG_Normalized',
            'Heading',
            'Step_Time',
            'Step_Distance',
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


    # VALIDITY -----------------------------------------------------------------
    def plot_hist(self, column):
        '''Plot time lag.'''
        plt.style.use(['ggplot'])
        fig = plt.figure()
        plt.title('{0} Histogram'.format(column))
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.hist(self.df['column'], color='dodgerblue')
        plt.show()

        # df.df['Step_ROT'].hist(by=df.df['VesselType'])

    def step_sog(self):
        '''Caclulate the speed required for step distacnce.'''
        self.df['Step_SOG'] = 3600 * self.df['Step_Distance']/self.df['Step_Time']
        self.df['Error_SOG'] = self.df['Step_SOG'].divide(
            self.df['SOG'],
            fill_value = 0) - 1


    # HELPER FUNCTIONS ---------------------------------------------------------
    def normalize_angle(self, column, start, end):
        '''Normalized an angle to be within the start and end.'''
        width = end - start
        offset = self.df[column] - start
        name = '{0}_Normalized'.format(column)
        self.df[name] = offset - np.floor(offset/width)*width + start

    def normalize_angle_diff(self, column, start, end):
        '''Normalized an angle to be within the start and end.'''
        self.df['Difference'] = self.grouped_time[column].diff()
        width = end - start
        offset = self.df['Difference'] - start
        name = '{0}_Difference'.format(column)
        self.df[name] = offset - np.floor(offset/width)*width + start
        self.df.drop(columns=['Difference'], inplace=True)


@time_all
class Analysis_Dataframe(object):

    def __init__(self, df):
        self.df = df

    @property
    def grouped_track(self):
        return self.df.sort_values(
            ['mmsi', 'own_trackid', 'basedatetime']
        ).groupby(
            groupBy
        )

    def split_straight_maneuver(self):
        '''put a 1 if accel > 2 or if rot > 10'''
        # First do analysis of range of values
        return
