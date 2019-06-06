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
import geopy.distance
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import time_all, print_reduction


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
            'Heading',
            'VesselType'
        ]

        self.df = pd.read_csv(self.csv, usecols=self.headers)
        self.df['BaseDateTime'] = pd.to_datetime(self.df['BaseDateTime'])
        self.sortBy = ['MMSI', 'BaseDateTime']
        for col in ['VesselType', 'Cargo', 'Status']:
            self.df[col].astype('category')

    def clean(self):
        '''Remove bad data from dataframe.'''
        self.drop_spatial()
        self.drop_duplicate_keys()
        self.drop_bad_mmsi()
        self.drop_bad_speed()
        self.drop_bad_heading()
        self.drop_null()
        self.drop_sparse()

    # CLEAN DATA ---------------------------------------------------------------
    def sorted_group(self):
        return self.df.sort_values(self.sortBy).groupby('MMSI')

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
    def drop_duplicate_keys(self):
        '''MMSI, BaseDateTime are primary key, should be unique.'''
        self.df.drop_duplicates(
            subset=['MMSI', 'BaseDateTime'],
            keep=False,
            inplace=True
        )

    @print_reduction
    def drop_bad_mmsi(self):
        '''MMSI numbers should be 9 digits.'''
        cond = self.df['MMSI'].apply(lambda x: len(str(x)) == 9)
        self.df = self.df[cond].copy()

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
    def drop_sparse(self):
        '''Remove MMSIs that have less than 10 data points.'''
        self.df = self.df.groupby('MMSI').filter(lambda g: len(g) > 50)


    # SEGMENT TRAJECTORIES -----------------------------------------------------
    def time_interval(self):
        '''Add time between cleaned rows.'''
        self.df['Time_Interval'] = self.sorted_group()['BaseDateTime'].diff()
        self.df['Time_Interval'].fillna(pd.Timedelta(0), inplace=True)

    def mark_time_gap(self, maxJump='3.5m'):
        '''Mark points with large time gap.'''
        self.time_interval()
        self.df.rename(columns = {'Time_Interval': 'Time_Jump'}, inplace=True)
        condTime = self.df['Time_Jump'] > pd.Timedelta(maxJump)
        self.df['Time_Break'] = np.where(condTime, 1, 0)

    def group_time_gap(self):
        '''Group points that do not have a large time gap.'''
        self.df['Time_Track'] = self.sorted_group()['Time_Break'].cumsum()
        self.df.drop(columns=['Time_Break'], inplace=True)
        self.df['Time_Interval'] = self.df.sort_values(
            ['MMSI', 'Time_Track', 'BaseDateTime']
            ).groupby(['MMSI', 'TimeTrack'])['BaseDateTime'].diff()

    def relative_displacement(self):
        '''Calculate the relative start-to-date displacement between rows.'''
        pointFirst = self.sorted_group().head(1)[['MMSI', 'LAT', 'LON']].copy()
        pointFirst.rename(columns = {
            'LAT': 'LAT_First',
            'LON': 'LON_First'
        }, inplace=True)
        self.df = self.df.merge(pointFirst, on='MMSI')

        def distance(row):
            '''Return distance between row position and lag potition in nm.'''
            return geopy.distance.distance(
                (row['LAT'], row['LON']),
                (row['LAT_First'], row['LON_First'])
            ).nm

        self.df['Displaced'] = self.df.apply(distance, axis=1).pct_change(fill_method='bfill')

    def assign_stop(self, maxTime, minDisplace):
        '''Assign status 'stop' to a point if it satisfies criteria'''
        condTime = self.df['Time_Interval'] > maxTime
        condDisplace = self.df['Displaced'] < minDisplace
        self.df['Stop'] = np.where(condTime & condDisplace, 1, 0)


    def time_break(self, interval):
        '''
        Break a MMSI trajectory into subtrajectories based on time
        interval between points. Then get the time lags between points
        within subtrajectories.
        '''
        self.df['Time_Break'] = np.where(self.df['Time_Lag'] > interval, 1, 0)
        self.df['Time_Track'] = self.df.groupby('MMSI')['Time_Break'].cumsum()

    def time_plot(self):
        '''Plot time lag.'''
        fig = plt.figure()
        plt.subplot(211)
        plt.scatter(self.df['Time_Interval'].astype('timedelta64[s]'), self.df['SOG'])

        plt.subplot(212)
        plt.scatter(self.df['Time_Interval'].astype('timedelta64[s]'), self.df['Displaced'])
        plt.show()

    # Exploratory data analysis
    def eda(self, time_max):
        '''Exploratory Data Analysis.'''
        # Does calculated speed match reported
        self.df['SOG_Calculated'] = (
            self.df['Distance_Lag']/self.df['Time_Lag'])*(60*60)

    def add_track(self):
        '''Add track ID for each MMSI.'''
        self.df.sort_values(by=['MMSI', 'BaseDateTime'], inplace=True)

        self.df['Time_Delta'] = self.df.groupby('MMSI')['BaseDateTime'].diff()
        self.df['Time_Delta'].fillna(pd.Timedelta(seconds=0), inplace=True)

        # If time difference is greater than 3 minutes consider it new track
        timeMax = 3
        overMax = self.df['Time_Delta'] > datetime.timedelta(minutes=timeMax)
        self.df['Break'] = np.where(overMax, 1, 0)

        # Add incrementing track ID for each MMSI
        self.df['TrackID'] = self.df.groupby('MMSI')['Break'].cumsum()

        # Clean up
        self.df.drop(columns=['Break'], inplace=True)

    # Add rate of turn to each point in a track
    def add_ROT(self):
        '''Add ROT field.'''
        self.df.sort_values(by=['MMSI', 'TrackID', 'BaseDateTime'], inplace=True)
        cols = ['MMSI', 'TrackID']
        self.df['Heading_Diff'] = self.df.groupby(cols)['Heading'].diff()
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
            'BaseDateTime',
            'TrackID',
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


    #
    # # Distance helper functions
    # def position_lag(self, lag_max):
    #     '''Add columns with the lag position.'''
    #
    #     def dynamic_lag(df):
    #         '''Shift position by either the lag_max or the length of the df.'''
    #         # TODO: deal with sign
    #         if lag_max < 0:
    #             lag = max(lag_max, -len(df)+1)
    #         else:
    #             lag = min(lag_max, len(df)-1)
    #         df['LAT_Lag'] = df['LAT'].shift(lag).fillna(method='ffill')
    #         df['LON_Lag'] = df['LON'].shift(lag).fillna(method='ffill')
    #         return df
    #
    #     def distance(row):
    #         '''Return distance between row position and lag potition in nm.'''
    #         return geopy.distance.distance(
    #             (row['LAT'], row['LON']),
    #             (row['LAT_Lag'], row['LON_Lag']),
    #         ).nm
    #
    #
    #     self.df.sort_values(['MMSI', 'Time_Track', 'BaseDateTime'], inplace=True)
    #     self.df.groupby(['MMSI', 'Time_Track']).apply(dynamic_lag)
    #     self.df['Displacement_Lag'] = self.df.apply(distance, axis=1)
    #
    #

# def identify_stops(self, radius):
#
#     # Define clustering algorithm
#     def location(df, index):
#         '''Return the lat, long tuple at the index given.'''
#         return (df.iloc[index].at['LAT'], df.iloc[index].at['LON'])
#
#     def stop_check(df):
#         '''Cluster based on radius.'''
#         if len(df) < 50:
#             df['status'] = 'undefined'
#             return
#         df = df.copy().sort_values('BaseDateTime').reset_index()
#         df['status'] = 'undefined'
#         first_index = 0
#         for index, row in df.iterrows():
#             if index in range(first_index, first_index + 5):
#                 df.iloc[index].at['status'] = np.nan
#                 continue
#             # calculate the distance between the first and 5th rows
#             first = location(df, first_index)
#             distance = haversine(
#                 first,
#                 (row['LAT'], row['LON']),
#                 unit='nmi'
#             )
#             # if distance is greater than the radius, increment counter
#             if distance > radius:
#                 status = 'underway'
#                 first_index = index
#             else:
#                 status = 'stopped'
#             row['status'] = status
#
#     # Apply clustering algorithm
#     df = self.df.groupby('MMSI').apply(stop_check)
#
#     # Replace 'stopped' with center point
