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
import angles
import datetime
import geopandas
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
from os.path import basename, dirname, join
import pandas as pd
from pandas.plotting import register_matplotlib_converters
import seaborn as sns

from . import check_length, print_reduction, time_all
from . import concat_df, snap_value, buffer_bounds

# sns.set_palette("Set1")
# sns.set_context("paper", font_scale=3)
sns.axes_style("darkgrid")

register_matplotlib_converters()
plt.rcParams.update({'font.size': 6})

# ------------------------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------------------------
METERS_IN_NM = 1852
EARTH_RADIUS_KM = 6371
DIRECTORY_PLOTS = 'D:\\Maura\\Plots\\2017'
V_TYPES = {
    30: 'fishing',
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
    1001: 'fishing',
    1002: 'fishing',
    1003: 'cargo',
    1004: 'cargo',
    1012: 'ferry',
    1013: 'passenger',
    1014: 'passenger',
    1016: 'cargo',
    1017: 'tanker',
    1019: 'recreational',
    1023: 'tug',
    1024: 'tanker',
    1025: 'tug'
}


# ------------------------------------------------------------------------------
# FUNCTIONS
# ------------------------------------------------------------------------------
def haversine(lat1, lon1, lat2, lon2):
    '''Return the haversine distance between two points.'''
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
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

def bearing(lat1, lon1, lat2, lon2):
    '''Return the haversine distance between two points.'''
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    return angles.r2d(angles.bear(lon1, lat1, lon2, lat2))


# ------------------------------------------------------------------------------
# DATAFRAMES
# ------------------------------------------------------------------------------
@time_all
class BS_Dataframe(object):

    def __init__(self):

        bs_data = [
            (42.440139, -124.416889),
            (43.274167, -124.37),
            (45.739417, -124.372472),
            (46.163739, -122.852656),
            (46.262222, -123.888417),
            (47.549083, -122.784639),
            (47.615833, -122.308889),
            (48.370667, -124.672972),
            (48.453889, -122.625833)
            ]
        self.df = pd.DataFrame(bs_data, columns=['LAT', 'LON'])
        self.gdf = geopandas.GeoDataFrame(
            self.df,
            geometry=geopandas.points_from_xy(self.df['LON'], self.df['LAT']),
            crs={'init':'epsg:4326'})
        self.gdf = self.gdf.to_crs(epsg=32610)




@time_all
class EDA_Dataframe(object):

    '''
    Explore the NAIS data.
    '''

    def __init__(self, csvFiles, lonMin, lonMax, latMin, latMax):
        '''Process nais dataframe.'''
        self.csvs = csvFiles

        # Spatial parameters
        self.lonMin = lonMin
        self.lonMax = lonMax
        self.latMin = latMin
        self.latMax = latMax

        # Spatial reference id
        self.srid = 32610

        # Buffer on area of interest - location specific
        self.offset = 0.0254

        # Base stations
        bs_data = [
            (42.440139, -124.416889),
            (43.274167, -124.37),
            (45.739417, -124.372472),
            (46.163739, -122.852656),
            (46.262222, -123.888417),
            (47.549083, -122.784639),
            (47.615833, -122.308889),
            (48.370667, -124.672972),
            (48.453889, -122.625833)
        ]
        self.bs = pd.DataFrame(bs_data, columns=['LAT', 'LON'])

        # Create raw NAIS dataframe
        self.df = concat_df(pd.read_csv, self.csvs)
        self.df['BaseDateTime'] = pd.to_datetime(self.df['BaseDateTime'])

        # Categorize status
        self.df['Status'].replace(np.nan, "undefined", inplace=True)
        self.df['Status'] = self.df['Status'].astype('category')

        self.required = ['MMSI', 'BaseDateTime', 'LAT', 'LON', 'SOG', 'COG']


    # PROPERTIES ---------------------------------------------------------------
    @property
    def grouped_mmsi(self):
        '''Return sorted dataframe grouped by MMSI.'''
        return self.df.sort_values(['MMSI', 'BaseDateTime']).groupby('MMSI')

    @property
    def grouped_trajectory(self):
        '''Return sorted dataframe grouped by MMSI and Time Track.'''
        return self.df.sort_values(
            ['MMSI', 'Trajectory', 'BaseDateTime']
        ).groupby(
            ['MMSI', 'Trajectory']
        )


    # MAIN FUNCTIONS -----------------------------------------------------------
    @print_reduction
    def clean_raw(self):
        '''
        Select the area of interest, remove duplicate information, remove
        contradictory information, remove invalid IDs.
        '''
        # Area of interest
        self.select_spatial()
        # No loss of information
        self.drop_null()
        self.drop_duplicate_rows()
        # Remove contradictory data
        self.drop_duplicate_keys()
        self.drop_inconsistent_info()
        # Remove invalid and insufficient IDs
        self.drop_bad_mmsi()
        self.drop_single_mmsi()
        # Standardize data and sort
        self.map_vessel_types()
        self.normalize_cog()
        self.df.sort_values(['MMSI', 'BaseDateTime'], inplace=True)

    def step_calculations(self):
        '''
        Calculate the time, distance, sog, and cog between successive points
        within each MMSI.
        '''
        self.step_time()
        self.step_distance()
        self.step_sog()
        self.step_cog()
        # self.mmsi_plot()


    def remove_stops(self):
        '''
        Remove data that has a time interval of more than 2 minutes
        '''
        self.step_time()
        self.drop_stop()
        self.step_time()

    def make_trajectories(self):
        '''
        ID where ship's enter and exit bounding box.
        '''



        #
        # self.mark_enter_exit()
        #
        # self.mark_jump()
        # self.distance_to_bound()
        # self.distance_to_bs()
        # self.step_distance()
        # self.step_sog()

        # self.average_boundary_step()
        # Create GeoDataFrame of clean nais data
        # self.create_gdf()


    @print_reduction
    def clean_gdf(self, sensitivity=0.15, maxJump=4):
        '''Clean suspicious data from geodataframe.'''


        #
        self.step_distance()
        # self.step_sog()



        # self.drop_bad_speed()
        # self.drop_bad_heading()
        # self.drop_vessel_types()
        #
        #
        #
        # # add check columns
        # self.point_cog()
        #
        #
        # # drop suspicious data
        # self.drop_bad_distance(sensitivity)
        # self.mark_time_jump(maxJump)
        # self.drop_jump_string()

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

    def summary_plots(self):
        '''Save plots from the dataframe.'''
        dir = 'D:\\Maura\\Plots\\2017'
        fig = plt.figure()
        plt.xlabel('Vessel Type')
        sns.countplot('VesselType', data = self.df)
        plt.savefig(join(dir,'Vessel_Type.png'))

        fig = plt.figure()
        plt.xlabel('SOG')
        self.df['SOG'].hist()
        plt.savefig(join(dir,'SOG.png'))

        sns.lmplot('Step_Distance','Expected_Distance',data=self.df)
        plt.savefig(join(dir,'Point_V_Speed_Distance.png'))

        sns.lmplot('COG','Heading',data=self.df)
        plt.savefig(join(dir,'COG_V_Heading.png'))

    def save_output(self):
        '''Save output to be read into PostgreSQL.'''
        self.validate_types()
        self.normalize_time()
        self.reorder_output()


    # DATAFRAME CLEANING -------------------------------------------------------
    @print_reduction
    def select_spatial(self):
        '''
        Limit data to bounding box of interest plus offset. Add flag if row
        is inside bounding box. This allows step calculations to be calculated
        at boundary without generating weird values. Adds a field to dataframe.
        '''
        # Select points within the buffered area of interest
        lonMinB, lonMaxB = self.lonMin-self.offset, self.lonMax-self.offset
        lonMinB, lonMaxB = buffer_bounds(self.lonMin, self.lonMax, self.offset)
        latMinB, latMaxB = buffer_bounds(self.latMin, self.latMax, self.offset)

        self.df = self.df[
            (self.df['LON'].between(lonMinB, lonMaxB)) &
            (self.df['LAT'].between(latMinB, latMaxB))
        ].copy()

        # Identify points in the boundary and those in the buffer
        self.df['In_Bound'] = np.where(
            (self.df['LON'].between(self.lonMin, self.lonMax)) &
            (self.df['LAT'].between(self.latMin, self.latMax)),
            1,
            0
        )

    @print_reduction
    def drop_null(self):
        '''Drop rows with nulls in the required columns (omitted data).'''
        self.df.replace("", np.nan, inplace=True)
        self.df.dropna(how='any', subset=self.required, inplace=True)

    @print_reduction
    def drop_duplicate_rows(self):
        '''Remove entirely duplicated rows. No loss of information.'''
        self.df.drop_duplicates(keep='first', inplace=True)

    @print_reduction
    def drop_duplicate_keys(self):
        '''
        MMSI, BaseDateTime must be unique. Ship can't be in two places at once.
        Can't calculate step calculations with duplicate timestamps.
        Drop both duplicate rows.
        '''
        key = ['MMSI', 'BaseDateTime']

        # Save duplicates for further investigation
        mmsi_duplicate = self.df[self.df.duplicated(subset=key)]['MMSI'].unique()
        self.duplicates = self.df[self.df['MMSI'].isin(mmsi_duplicate)]

        # Drop duplicates from dataframe
        self.df.drop_duplicates(subset=key, keep=False, inplace=True)

    @print_reduction
    def drop_inconsistent_info(self):
        '''
        Confirm that a MMSI is associated with only one name, dimension.
        A mix of vessels using the same MMSI will not be included.
        This data is entered only once, so a mistake in entry will appear
        in all data points and not change over time.
        '''
        mmsi = self.df.groupby(['MMSI'])
        self.df = mmsi.filter(lambda g: g['VesselName'].nunique()<=1)
        self.df = mmsi.filter(lambda g: g['Length'].nunique()<=1)
        self.df = mmsi.filter(lambda g: g['Width'].nunique()<=1)

    @print_reduction
    def drop_bad_mmsi(self):
        '''MMSI numbers should be 9 digits and between a given range.'''
        condRange = self.df['MMSI'].between(201000000, 775999999)

        # Save bad data for further investigation
        self.bad_mmsi = self.df[~condRange].copy()

        # Drop bad MMSIs from data
        self.df = self.df[condRange]

    @print_reduction
    def drop_single_mmsi(self):
        '''
        Remove MMSIs with only one row.
        '''
        self.single = self.df.groupby(['MMSI']).filter(lambda g: len(g)==1)
        self.df = self.df.groupby(['MMSI']).filter(lambda g: len(g)>1)

    def normalize_cog(self):
        '''Normalize COG to an angle between [0, 360).'''
        self.df['COG'] = self.df['COG'].apply(
            lambda x: angles.normalize(x, 0, 360)
        )

    def map_vessel_types(self):
        '''Map codes to categories.'''
        self.df['VesselType'].replace("", np.nan, inplace=True)
        self.df['VesselType'] = self.df['VesselType'].map(V_TYPES)
        self.df['VesselType'] = self.df['VesselType'].astype('category')


    # STEP FUNCTIONS -----------------------------------------------------------
    def step_time(self):
        '''Return time between timestamps. Adds a field to dataframe.'''
        col = 'Time Interval'
        self.df[col] = self.grouped_mmsi['BaseDateTime'].diff()
        self.df[col].fillna(datetime.timedelta(seconds=60), inplace=True)
        self.df[col] = self.df[col].astype('timedelta64[s]')

    def step_distance(self):
        '''Return distance between lat/lon positions.'''
        def distance(df):
            df.reset_index(inplace=True)
            df['Displacement'] = haversine(
                df['LAT'].shift(),
                df['LON'].shift(),
                df.loc[1:,'LAT'],
                df.loc[1:,'LON'])
            return df.set_index('index')
        self.df = self.grouped_mmsi.apply(distance)

    def step_sog(self):
        '''Caclulate the speed required for step distacnce.'''
        self.df['Interval SOG'] = (
            3600*self.df['Displacement']/self.df['Time Interval'])

        def rolling(df):
            df.reset_index(inplace=True)
            df['Segment SOG'] = df['SOG'].abs().rolling(window=2).mean()
            return df.set_index('index')
        self.df = self.grouped_mmsi.apply(rolling)
        self.df['SOG Difference'] = abs(self.df['Interval SOG']) - self.df['Segment SOG']

        # fig = plt.figure()
        # sns.lmplot('Segment SOG','Interval SOG', data=self.df)
        # plt.savefig(join(DIRECTORY_PLOTS,'SOG_raw.png'))
        # plt.close()
        #
        # # sns.lmplot('Step_SOG', 'SOG Difference', data=plot_df)
        # # plt.savefig(join(DIRECTORY_PLOTS,'SOG_Diff.png'))
        # fig = plt.figure()
        # sns.lmplot('Time Interval', 'SOG Difference', data=self.df)
        # plt.savefig(join(DIRECTORY_PLOTS,'SOG_Time.png'))
        # plt.close()

    def step_cog(self):
        '''Calculate the course between two position points.'''
        def course(df):
            df.reset_index(inplace=True)
            df['Interval COG'] = azimuth(
                df['LAT'].shift(),
                df['LON'].shift(),
                df.loc[1:,'LAT'],
                df.loc[1:,'LON']
            )
            return df.set_index('index')

        # Calculate and normalize course between successive points
        self.df = self.grouped_mmsi.apply(course)
        self.df['Interval COG'].fillna(method='bfill', inplace=True)
        self.df['Interval COG'] = self.df['Interval COG'].apply(
            lambda x: angles.normalize(x, 0, 360)
        )

        # Caclulate error
        self.df['Error_COG'] = (self.df['COG'] - self.df['Interval COG']) % 360

    def mmsi_plot(self):
        '''Plot raw trajectory for each MMSI.'''
        for name, group in self.grouped_mmsi:
            try:
                mmsi = group['MMSI'].unique()[0]
                print('Plotting MMSI %s' % mmsi)
                data = [
                    ['LAT'],
                    ['LON'],
                    ['SOG','Interval SOG', 'Segment SOG'],
                    ['COG', 'Interval COG'],
                    ['Time Interval']
                ]
                legend = [False, False, 'full', 'full', False]
                linestyles = ["","-", "--"]

                # Create plots
                fig, axes = plt.subplots(5, 1, sharex='col', figsize=(8, 11))
                fig.suptitle("MMSI: {0}".format(mmsi), fontsize=14)
                for i in range(5):
                    for j in range(len(data[i])):
                        sns.lineplot(
                            x='BaseDateTime',
                            y=data[i][j],
                            style=linestyle[j],
                            data=group,
                            palette=sns.color_palette("paired", len(data[i])),
                            ax=axes[i],
                            label=data[i][j],
                            legend=legend[i]
                        )
                        axes[i].set_ylabel(data[i][0])

                # Format plot area
                plt.xticks(rotation=70)
                plt.tight_layout()
                fig.subplots_adjust(top=0.9)

                # Save to directory
                plt.savefig(
                    join(DIRECTORY_PLOTS, 'MMSI', 'Raw', '{0}.png'.format(mmsi))
                )
                plt.close()
            except:
                continue


    # FIND ERRONEOUS DATA ------------------------------------------------------
    @print_reduction
    def drop_stop(self, maxTime=2):
        '''Assign status 'stop' to a point if it satisfies criteria'''
        # Transmission frequency over 2 minutes
        cond_time = (self.df['Time Interval'] > maxTime*60)
        cond_speed = (self.df['SOG'] == 0)
        self.df['Stop'] = np.where(cond_time | cond_speed, 1, 0)

        # Status is stopped
        status_stop = ['not under command', 'at anchor', 'moored', 'aground']
        self.df['Stop'] = np.where(
            self.df['Status'].isin(status_stop),
            1,
            self.df['Stop']
        )

        self.stops = self.df[self.df['Stop']==1]
        self.df = self.df[self.df['Stop']==0]



    # def mmsi_plot(self):
    #     '''Plot raw trajectory for each MMSI.'''
    #     for name, group in self.grouped_mmsi:
    #         mmsi = group['MMSI'].unique()[0]
    #         data = ['LAT', 'LON', 'SOG', 'COG', 'Time Interval']
    #         # legend = ['full', False, False, False, False]
    #         legend = [False, False, False, False, False]
    #
    #         # Create plots
    #         fig, axes = plt.subplots(5, 1, sharex='col', figsize=(8, 11))
    #         fig.suptitle("MMSI: {0}".format(mmsi), fontsize=14)
    #         for i in range(0,5):
    #             axes[i].set_ylabel(data[i])
    #             sns.lineplot(
    #                 x='BaseDateTime',
    #                 y=data[i],
    #                 # hue='In_Bound',
    #                 data=group,
    #                 # palette=sns.color_palette("cubehelix", 2),
    #                 ax=axes[i],
    #                 legend=legend[i]
    #             )
    #
    #         # First plot with legend
    #         # axes[0].legend(
    #         #     loc='upper center',
    #         #     bbox_to_anchor=(0.5, 1.35),
    #         #     fancybox=True)
    #
    #         # Format plot area
    #         plt.xticks(rotation=70)
    #         plt.tight_layout()
    #         fig.subplots_adjust(top=0.95)
    #
    #         # Save to directory
    #         plt.savefig(
    #             join(DIRECTORY_PLOTS, 'MMSI', 'Raw', '{0}.png'.format(mmsi))
    #         )
    #         plt.close()
    #










    def bearing(self):
        def course(df):
            df.reset_index(inplace=True)
            df['Bearing'] = bearing(
                df['LAT'].shift(),
                df['LON'].shift(),
                df.loc[1:,'LAT'],
                df.loc[1:,'LON']
            )
            return df.set_index('index')
        self.df = self.grouped_mmsi.apply(course)
        self.df['Bearing'].fillna(method='bfill', inplace=True)




    # SUB-TRAJECTORIES ---------------------------------------------------------
    # def mark_enter_exit(self):
    #     '''
    #     Mark points that are exits and entrances to area. Adds a field
    #     to dataframe.
    #     '''
    #     def enter_exit(df):
    #         df.reset_index(inplace=True)
    #         switch = df['In_Bound'].diff()
    #         df['In_Out'] = np.where(
    #             (switch==1) | (switch==-1), 1, 0)
    #         return df.set_index('index')
    #     self.df = self.grouped_mmsi.apply(enter_exit)
    #     self.ee = self.df[self.df['In_Out']==1]
    #
    #     self.df['Trajectory'] = self.grouped_mmsi['Break'].cumsum()

    def mark_jump(self):
        '''Mark points with large time jump. Adds a field to dataframe.'''
        cutoff = self.df['Time Interval'].quantile(0.99)
        self.df['Jump'] = np.where(
            (self.df['Time Interval']>cutoff) & (self.df['In_Bound']==1),
            1,
            0
        )
        self.df['Break'] = np.where(
            (self.df['Jump']==1) | (self.df['In_Out']==1),
            1,
            0
        )
        self.df['Trajectory'] = self.grouped_mmsi['Break'].cumsum()

    @print_reduction
    def drop_single_trajectory(self):
        '''
        Remove trajectories with only one row.
        '''
        self.df = self.grouped_trajectory.filter(lambda g: len(g)>1)
        self.df['Trajectory'] = self.grouped_mmsi['Break'].cumsum()

    # def mmsi_plot(self):
    #     '''Plot raw trajectory for each MMSI.'''
    #     # plt.style.use('dark_background')
    #     for name, group in self.grouped_mmsi:
    #         mmsi = group['MMSI'].unique()[0]
    #         data = ['LAT', 'LON', 'SOG', 'COG', 'Time Interval']
    #         legend = ['full', False, False, False, False]
    #         ntracks = len(group['Trajectory'].unique())
    #         # Create plots
    #         fig, axes = plt.subplots(5, 1, sharex='col', figsize=(8, 11))
    #         fig.suptitle("MMSI: {0}".format(mmsi), fontsize=14)
    #         for i in range(0,5):
    #             axes[i].set_ylabel(data[i])
    #             sns.lineplot(
    #                 x='BaseDateTime',
    #                 y=data[i],
    #                 hue='Trajectory',
    #                 data=group,
    #                 palette=sns.color_palette("cubehelix", ntracks),
    #                 ax=axes[i],
    #                 legend=legend[i]
    #             )
    #
    #         # First plot with legend
    #         axes[0].legend(
    #             loc='upper center',
    #             bbox_to_anchor=(0.5, 1.35),
    #             ncol=12,
    #             fancybox=True)
    #
    #         # Format plot area
    #         plt.xticks(rotation=70)
    #         plt.tight_layout()
    #         fig.subplots_adjust(top=0.90)
    #
    #         # Save to directory
    #         plt.savefig(
    #             join(DIRECTORY_PLOTS, 'MMSI', 'Raw', '{0}.png'.format(mmsi))
    #         )
    #         plt.close()




    # def step_distance(self):
    #     '''Return distance between timestamps.'''
    #     def distance(df):
    #         df.reset_index(inplace=True)
    #         df['Step_Distance'] = haversine(
    #             df['LAT'].shift(),
    #             df['LON'].shift(),
    #             df.loc[1:,'LAT'],
    #             df.loc[1:,'LON'])
    #         return df.set_index('index')
    #
    #     # Distance based on difference in position
    #     self.df = self.grouped_trajectory.apply(distance)



    def distance_to_bs(self):
        '''Find minimum distance from point to base stations.'''
        for i in range(len(self.bs)):
            name = 'Distance_BS{0}'.format(str(i))
            nameLat = 'LAT_BS{0}'.format(str(i))
            nameLon = 'LON_BS{0}'.format(str(i))
            self.df[nameLat] = self.bs.iloc[i].at['LAT']
            self.df[nameLon] = self.bs.iloc[i].at['LON']
            self.df[name] = haversine(
                self.df['LAT'],
                self.df['LON'],
                self.df[nameLat],
                self.df[nameLon]
            )
            self.df.drop(columns=[nameLat, nameLon], inplace=True)
        bs_cols = [col for col in self.df.columns if 'Distance_BS' in col]
        self.df['Distance_BS'] = self.df[bs_cols].min(axis=1)
        self.df.drop(columns=bs_cols, inplace=True)


        # plot_df = self.df[(self.df['In_Bound']==1) & (~self.df['Step_SOG'].isnull()) & (self.df['Jump']==0)]
        # fig, ax = plt.subplots()
        # for a in ['SOG', 'Segment_SOG', 'Step_SOG']:
        #     sns.distplot(plot_df[a])
        # plt.savefig(join(DIRECTORY_PLOTS,'SOG_hist.png'))
        #
        # fig = plt.figure()
        # sns.lmplot('Segment_SOG','Step_SOG', data=plot_df)
        # plt.savefig(join(DIRECTORY_PLOTS,'SOG_raw.png'))
        #
        # sns.lmplot('Step_SOG', 'Diff_SOG', data=plot_df)
        # plt.savefig(join(DIRECTORY_PLOTS,'SOG_Diff.png'))
        #
        # sns.lmplot('Time Interval', 'Diff_SOG', data=plot_df)
        # plt.savefig(join(DIRECTORY_PLOTS,'SOG_Time.png'))

        # self.df['Error_SOG'] = (
        #     self.df['Step_SOG'].divide(self.df['Segment_SOG'], fill_value = 0) - 1
        # )


        # sns.lmplot('Segment_SOG','Step_SOG', data=self.df[self.df['Error_SOG'].abs()<=.05])
        # plt.savefig(join(DIRECTORY_PLOTS,'SOG5_processed.png'))
        #
        # self.df['Error_SOG'].hist()
        # plt.savefig(join(DIRECTORY_PLOTS,'SOG_error.png'))





    def distance_to_bound(self):
        '''
        Find minimum distance between point and bounding box.
        Adds a field to dataframe.
        '''
        self.df['NS_Lat'] = self.df['LAT'].apply(
            lambda x: snap_value(x, self.latMin, self.latMax))
        self.df['EW_Lon'] = self.df['LON'].apply(
            lambda x: snap_value(x, self.lonMin, self.lonMax))

        self.df['NS'] = haversine(
            self.df['LAT'],
            self.df['LON'],
            self.df['NS_Lat'],
            self.df['LON'])
        self.df['EW'] = haversine(
            self.df['LAT'],
            self.df['LON'],
            self.df['LAT'],
            self.df['EW_Lon'])
        self.df['Distance_BB'] = self.df[['NS','EW']].min(axis=1)
        self.df.drop(columns=['NS_Lat', 'EW_Lon', 'NS', 'EW'], inplace=True)



    def average_boundary_step(self):
        '''
        Mark points that are exits and entrances to area.
        Get average step size.
        '''
        def enter_exit(df):
            df.reset_index(inplace=True)
            df['In_Out'] = np.where(
                (df['Inside_Boundary'].diff() == 1) |
                (df['Inside_Boundary'].diff().shift() == -1) &
                (df['Step_Time'] < 1.5*60),
                1,
                0
            )
            return df.set_index('index')
        self.df = self.grouped_mmsi.apply(enter_exit)
        self.ee = self.df[self.df['Enter_Exit']==1]

        fig = plt.figure()
        sns.lmplot('Distance_BB','SOG', data=self.df)
        plt.show()





    def fill_defaults(self):
        '''Interpolate over single default values.'''
        self.df['Default_SC'] = np.where(
            ((self.df['SOG']==-0.1) | (self.df['SOG']==0)) &
            (self.df['COG']==-49.6),
            1,
            0)

        # Flag single default values
        def switch(df):
            df.reset_index(inplace=True)
            df.sort_values(['BaseDateTime'], inplace=True)
            switch = df['Default_SC'].diff()
            df['Single_Default'] = np.where(
                (switch==1) & (switch.shift()==-1), 1, 0)
            return df.set_index('index')
        self.df = self.grouped_mmsi.apply(switch)

        # Interpolate single default values
        self.df.set_index('BaseDateTime', inplace=True)
        self.df['SOG'] = np.where(self.df['Single_Default']==1, np.nan, self.df['SOG'])
        self.df['COG'] = np.where(self.df['Single_Default']==1,  np.nan, self.df['COG'])
        self.df['SOG'].interpolate(method='time', limit=1, inplace=True)
        self.df['COG'].interpolate(method='time', limit=1, inplace=True)
        self.df.reset_index(inplace=True)
            # TODO: still need to handle consecutive


        # self.df['Error_SOG'] = np.where()
        #
        #


    def create_gdf(self):
        self.gdf = geopandas.GeoDataFrame(
            self.df,
            geometry=geopandas.points_from_xy(self.df['LON'], self.df['LAT']),
            crs={'init':'epsg:4326'})
        self.gdf = self.gdf.to_crs(epsg=self.srid)





    # def mark_enter_exit(self):

    # def step_distance(self):
    #     '''Return distance between timestamps.'''
    #     def distance(df):
    #         df.reset_index(inplace=True)
    #         df1 = df['geometry'].shift()
    #         df2 = df.loc[1:,'geometry']
    #         df['Step_Distance'] = df1.distance(df2)
    #         return df.set_index('index')
    #
    #     # Distance based on difference in position
    #     self.gdf = self.gdf.sort_values(['MMSI', 'BaseDateTime']).groupby('MMSI').apply(distance)




    # @print_reduction
    # def drop_jump_string(self):
    #     '''Drop consecutive time jumps.'''
    #     # if the current row Jump=1 and the next row Jump=0, its a track start
    #     def track_starts(df):
    #         df.reset_index(inplace=True)
    #         df['Track_Start'] = np.where(
    #             (self.df['Jump']==1) & (self.df['Jump'].shift()==0), 1, 0
    #         )
    #         df['Track_End'] = np.where(
    #             (self.df['Jump']==0) & (self.df['Jump'].shift(1)==1), 1, 0
    #         )
    #         return df.set_index('index')
    #
    #     self.df = self.grouped_mmsi.apply(track_starts)

        # Delete time jumps that are not the start of a track
        # jumps = self.df[(self.df['Jump']==1) & (self.df['Track_Start']==0)].index
        # self.df.drop(jumps , inplace=True)


    # GEODATAFRAME CLEANING ----------------------------------------------------
    # @print_reduction
    # def drop_bad_speed(self):
    #     self.df = self.df[self.df['SOG'] <= 102.2].copy()

    # DROP SUSPECT DATA





        # Distance based on speed and time interval
        # self.gdf['Expected_Distance'] = self.gdf['SOG']*self.gdf['Step_Time']/3600

        # self.df['Error_Distance'] = (
        #     self.df['Expected_Distance'].divide(self.df['Step_Distance'], fill_value = 0) - 1
        # )













    # @print_reduction
    # def drop_bad_speed(self):
    #     '''SOG should be positive.'''
    #     # What happens it I assume its positive?
    #     bad_mmsi = self.df[self.df['SOG'] <= -0.1]['MMSI'].unique().tolist()
    #     self.df = self.df[~self.df['MMSI'].isin(bad_mmsi)].copy()



    # FILTER DATA --------------------------------------------------------------
    def bad_mmsi_analysis(self):

        group = self.bad_mmsi.groupby('MMSI')
        self.runs=pd.DataFrame({
            'MMSI' : group['MMSI'].first(),
            'No. of Records' : group.size(),
            'Average Speed': group['SOG'].mean(),
            'VesselType' : group['VesselType'].first(),
            'Status' : group['Status'].first()
        }).reset_index(drop=True)
        # Make plot of locations





    @print_reduction
    def drop_vessel_types(self):
        '''Map codes to categories.'''
        self.df['VesselType'].replace("", np.nan, inplace=True)
        self.df.dropna(subset=['VesselType'], inplace=True)
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





    @print_reduction
    def drop_bad_distance(self, sensitivity):
        '''Drop distances outside expected values.'''
        # Plots
        fig = plt.figure()
        sns.lmplot('Expected_Distance','Step_Distance', data=self.df)
        plt.savefig(join(DIRECTORY_PLOTS,'Distance_raw.png'))

        sns.lmplot(
            'Expected_Distance',
            'Step_Distance',
            data=self.df[self.df['Step_Distance']<=1.15*self.df['Expected_Distance']]
        )
        plt.savefig(join(DIRECTORY_PLOTS,'Distance15_processed.png'))

        sns.lmplot(
            'Expected_Distance',
            'Step_Distance',
            data=self.df[self.df['Step_Distance']<=1.05*self.df['Expected_Distance']]
        )
        plt.savefig(join(DIRECTORY_PLOTS,'Distance5_processed.png'))

        self.df['Error_Distance'].hist()
        plt.savefig(join(DIRECTORY_PLOTS,'Distance_error.png'))


        max = (1 + sensitivity)*self.df['Expected_Distance']
        cond_distance = self.df['Step_Distance'] > max
        # At infrequent time intervals, speed is ~0, distance is less reliable
        cond_time = self.df['Step_Time'] < 120
        self.df['Drop'] = np.where((cond_distance) & (cond_time), 1, 0)

        self.df = self.df[self.df['Drop']==0].copy()
        self.df.drop(columns=['Drop'], inplace=True)




    # STOPS --------------------------------------------------------------------
    # def mark_stop(self, maxTime=2):
    #     '''Assign status 'stop' to a point if it satisfies criteria'''
    #     # Transmission frequency over 2 minutes
    #     cond_time = (self.df['Step_Time'] > maxTime*60)
    #     cond_speed = (self.df['SOG'] == 0)
    #     self.df['Stop'] = np.where(cond_time | cond_speed, 1, 0)
    #
    #     # Status is stopped
    #     status_stop = ['not under command', 'at anchor', 'moored', 'aground']
    #     self.df['Stop'] = np.where(
    #         self.df['Status'].isin(status_stop),
    #         1,
    #         self.df['Stop']
    #     )

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

    # def step_cog(self):
    #     '''Calculate change in course.'''
    #     def delta_course(df):
    #         df.reset_index(inplace=True)
    #         df['Step_COG_Radians'] = angle_difference(
    #             df['Point_COG'].shift(),
    #             df.loc[1:,'Point_COG']
    #         )
    #         return df.set_index('index')
    #     self.df = self.grouped_time.apply(delta_course)
    #     self.df['Step_COG_Radians'].fillna(method='bfill', inplace=True)
    #     self.df['COG_Cosine'] = np.cos(self.df['Step_COG_Radians'])
    #     self.df['Step_COG_Degrees'] = np.degrees(self.df['Step_COG_Radians'])


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
            'Trajectory',
            'LAT',
            'LON',
            'SOG',
            'COG',
            'VesselType'
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
            'COG'
        ]
        self.df = concat_df(pd.read_csv, self.csvs)
        # self.df = concat_df(pd.read_csv, self.csvs, usecols=self.headers)
        self.df['BaseDateTime'] = pd.to_datetime(self.df['BaseDateTime'])

        self.df['Status'].replace(np.nan, "undefined", inplace=True)
        self.df['Status'] = self.df['Status'].astype('category')

        # Normalize COG
        # self.normalize_angle('COG', 0, 360)
        # self.df.drop(columns='COG', inplace=True)
        # self.df.rename(columns={'COG_Normalized':'COG'}, inplace=True)


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
    def simplify(self):
        '''Select the area of interest and remove duplicate information.'''
        self.select_spatial()
        self.drop_null()
        self.drop_duplicate_rows()

        self.drop_duplicate_keys()
        self.drop_inconsistent_info()
        self.gdf = geopandas.GeoDataFrame(
            self.df,
            geometry=geopandas.points_from_xy(self.df['LON'], self.df['LAT']))

    @print_reduction
    def clean(self, sensitivity=0.15, maxJump=4):
        '''Clean bad or suspicious data from dataframe.'''

        self.drop_bad_mmsi()

    def reality_check(self):

        self.step_time()



        # DROP SUSPECT DATA

        self.step_distance()
        self.step_sog()

        # self.expected_distance()
        #
        # self.drop_bad_speed()
        # self.drop_bad_heading()
        # self.drop_vessel_types()
        #
        #
        #
        # # add check columns
        # self.point_cog()
        #
        #
        # # drop suspicious data
        # self.drop_bad_distance(sensitivity)
        # self.mark_time_jump(maxJump)
        # self.drop_jump_string()

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

    def plots(self):
        '''Save plots from the dataframe.'''
        dir = 'D:\\Maura\\Plots\\2017'
        fig = plt.figure()
        plt.xlabel('Vessel Type')
        sns.countplot('VesselType', data = self.df)
        plt.savefig(join(dir,'Vessel_Type.png'))

        fig = plt.figure()
        plt.xlabel('SOG')
        self.df['SOG'].hist()
        plt.savefig(join(dir,'SOG.png'))

        sns.lmplot('Step_Distance','Expected_Distance',data=self.df)
        plt.savefig(join(dir,'Point_V_Speed_Distance.png'))

        sns.lmplot('COG','Heading',data=self.df)
        plt.savefig(join(dir,'COG_V_Heading.png'))

    def save_output(self):
        '''Save output to be read into PostgreSQL.'''
        self.validate_types()
        self.normalize_time()
        self.reorder_output()

    # UTILITY FUNCTIONS --------------------------------------------------------
    @print_reduction
    def select_spatial(self):
        '''Limit to area of interest's bounding box.'''
        self.df = self.df[self.df['LON'].between(self.lonMin, self.lonMax)].copy()
        self.df = self.df[self.df['LAT'].between(self.latMin, self.latMax)].copy()

    @print_reduction
    def drop_null(self):
        '''Drop rows with nulls in the required columns (omitted data).'''
        for col in self.headers_required:
            self.df[col].replace("", np.nan, inplace=True)
        self.df.dropna(how='any', subset=self.headers_required, inplace=True)

    @print_reduction
    def drop_duplicate_rows(self):
        '''Remove entirely duplicated rows. No loss of information.'''
        self.df.drop_duplicates(keep='first', inplace=True)

    @print_reduction
    def drop_duplicate_keys(self):
        '''
        MMSI, BaseDateTime must be unique. Ship can't be in two places at once.
        Can't calculate step calculations with duplicate timestamps.
        Drop both duplicate rows.
        '''
        key = ['MMSI', 'BaseDateTime']

        # Save duplicates for further investigation
        self.duplicates = self.df[self.df.duplicated(subset=key)==True]

        # Drop duplicates from dataframe
        self.df.drop_duplicates(subset=key, keep=False, inplace=True)

    @print_reduction
    def drop_inconsistent_info(self):
        '''
        Confirm that a MMSI is associated with only one name, dimension.
        A mix of vessels using the same MMSI will not be included.
        '''
        mmsi = self.df.groupby(['MMSI'])
        self.df = mmsi.filter(lambda g: g['VesselName'].nunique()<=1)
        self.df = mmsi.filter(lambda g: g['Length'].nunique()<=1)
        self.df = mmsi.filter(lambda g: g['Width'].nunique()<=1)



    @print_reduction
    def drop_bad_mmsi(self):
        '''MMSI numbers should be 9 digits and between a given range.'''
        condRange = self.gdf['MMSI'].between(200999999, 776000000)

        # Save bad data for further investigation
        self.bad_mmsi = self.gdf[~condRange].copy()

        # Drop bad MMSIs from data
        self.gdf = self.gdf[condRange].copy()

    @print_reduction
    def drop_bad_speed(self):
        self.df = self.df[self.df['SOG'] <= 102.2].copy()

    # DROP SUSPECT DATA
    def step_time(self):
        '''Return time between timestamps.'''
        self.df['Step_Time'] = self.grouped_mmsi['BaseDateTime'].diff()
        self.df['Step_Time'] = self.df['Step_Time'].astype('timedelta64[s]')
        self.df['Step_Time'].fillna(method='bfill', inplace=True)

    def step_distance(self):
        '''Return distance between timestamps.'''
        # def distance(df):
        #     df.reset_index(inplace=True)
        #     df['Step_Distance'] = haversine(
        #         df['LAT'].shift(),
        #         df['LON'].shift(),
        #         df.loc[1:,'LAT'],
        #         df.loc[1:,'LON'])
        #     return df.set_index('index').fillna(method='bfill')
        def distance(df):
            df.reset_index(inplace=True)
            df['Step_Distance'] = df['geometry'].shift().distance(df.loc[1:,'geometry'])
            return df.set_index('index').fillna(method='bfill')

        # Distance based on difference in position
        self.gdf = self.grouped_mmsi.apply(distance)
        # self.df['Step_Distance'].fillna(method='bfill', inplace=True)

        # Distance based on speed and time interval
        self.gdf['Expected_Distance'] = self.gdf['SOG']*self.gdf['Step_Time']/3600

        # self.df['Error_Distance'] = (
        #     self.df['Expected_Distance'].divide(self.df['Step_Distance'], fill_value = 0) - 1
        # )






    def step_sog(self):
        '''Caclulate the speed required for step distacnce.'''
        self.df['Step_SOG'] = 3600 * self.df['Step_Distance']/self.df['Step_Time']
        def rolling(df):
            df.reset_index(inplace=True)
            df['Segment_SOG'] = df['SOG'].rolling(window=2).mean()
            return df.set_index('index')
        self.df = self.grouped_mmsi.apply(rolling)
        self.df['Segment_SOG'].fillna(method='bfill', inplace=True)

        self.df['Error_SOG'] = (
            self.df['Step_SOG'].divide(self.df['Segment_SOG'], fill_value = 0) - 1
        )


        fig = plt.figure()
        sns.lmplot('Segment_SOG','Step_SOG', data=self.df)
        plt.savefig(join(DIRECTORY_PLOTS,'SOG_raw.png'))

        sns.lmplot('Segment_SOG','Step_SOG', data=self.df[self.df['Error_SOG'].abs()<=.05])
        plt.savefig(join(DIRECTORY_PLOTS,'SOG5_processed.png'))

        self.df['Error_SOG'].hist()
        plt.savefig(join(DIRECTORY_PLOTS,'SOG_error.png'))






    # @print_reduction
    # def drop_bad_speed(self):
    #     '''SOG should be positive.'''
    #     # What happens it I assume its positive?
    #     bad_mmsi = self.df[self.df['SOG'] <= -0.1]['MMSI'].unique().tolist()
    #     self.df = self.df[~self.df['MMSI'].isin(bad_mmsi)].copy()



    # FILTER DATA --------------------------------------------------------------






    @print_reduction
    def drop_vessel_types(self):
        '''Map codes to categories.'''
        self.df['VesselType'].replace("", np.nan, inplace=True)
        self.df.dropna(subset=['VesselType'], inplace=True)
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





    @print_reduction
    def drop_bad_distance(self, sensitivity):
        '''Drop distances outside expected values.'''
        # Plots
        fig = plt.figure()
        sns.lmplot('Expected_Distance','Step_Distance', data=self.df)
        plt.savefig(join(DIRECTORY_PLOTS,'Distance_raw.png'))

        sns.lmplot(
            'Expected_Distance',
            'Step_Distance',
            data=self.df[self.df['Step_Distance']<=1.15*self.df['Expected_Distance']]
        )
        plt.savefig(join(DIRECTORY_PLOTS,'Distance15_processed.png'))

        sns.lmplot(
            'Expected_Distance',
            'Step_Distance',
            data=self.df[self.df['Step_Distance']<=1.05*self.df['Expected_Distance']]
        )
        plt.savefig(join(DIRECTORY_PLOTS,'Distance5_processed.png'))

        self.df['Error_Distance'].hist()
        plt.savefig(join(DIRECTORY_PLOTS,'Distance_error.png'))


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


def normalize_angle_diff(self, column, start, end):
    '''Normalized an angle to be within the start and end.'''
    self.df['Difference'] = self.grouped_mmsi[column].diff()
    width = end - start
    offset = self.df['Difference'] - start
    name = '{0}_Difference'.format(column)
    self.df[name] = offset - np.floor(offset/width)*width + start
    self.df.drop(columns=['Difference'], inplace=True)

#sns.boxplot(x="day", y="total_bill", hue="smoker", data=df, palette="Set1")

# '''Calculate change in course.'''
# self.normalize_angle_diff('Point_COG', 0, 360)
# self.df['Point_COG_Difference'].fillna(method='bfill', inplace=True)
# self.df.rename(columns={'Point_COG_Difference': 'Step_COG'}, inplace=True)
# self.df['COG_Cosine'] = np.cos(np.radians(self.df['Step_COG']))

# def plot_hist(self, column):
#     '''Plot time lag.'''
#     plt.style.use(['ggplot'])
#     fig = plt.figure()
#     plt.title('{0} Histogram'.format(column))
#     plt.xlabel(column)
#     plt.ylabel('Frequency')
#     plt.hist(self.df['column'], color='dodgerblue')
#     plt.show()
#
# df.df['Step_ROT'].hist(by=df.df['VesselType'])
# df.df['Stop'].hist(by=df.df['Status'])


# def check_duplicate(group):
#     if group['Duplicated'].sum()==0:
#         return group.drop(columns=['Duplicated'])
#
#     group.sort_values('BaseDateTime', inplace=True)
#     group.reset_index(inplace=True)
#     idx = group[group['Duplicated']==True].index
#     buffer = group.iloc[idx-1:idx+2]
#     return group.set_index('index')
