import gdal
import numpy as np
import pickle
import threading
from datetime import timedelta, date
from scipy.stats import linregress, gaussian_kde
from sklearn.covariance import EmpiricalCovariance
from sklearn.neighbors import BallTree, DistanceMetric
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from matplotlib import pyplot as plt
from matplotlib.dates import MonthLocator, DateFormatter
import matplotlib as mpl
mpl.rc("font", family="Helvetica")
mpl.rc("font", size=12)

site_name_abbr = {"Merced": "mb", "Tuolumne":"tb"}
density_factor = {"Merced": 0.333, "Tuolumne": 1.0}

class kNN_swe_regressor():
    def __init__(self, site_name, year, date_list, k):
        self.site_name = site_name
        self.year = year
        self.date_list = date_list[self.year][site_name]
        self.num_days = len(self.date_list)
        self.k = k
        self.kNN = None
        self.groundTruth = 'lidar'
        if self.year <= 2014:
            self.products = ['kNN', 'recon', 'snodas']
        else:
            self.products = ['kNN', 'snodas']
        self.est_dict = {'kNN':[], 'kNN_GP':[], 'lidar':[], 'recon':[], 'snodas': [], 'elev':[]}
        self.est_mean_dict = {'kNN':[], 'kNN_GP':[], 'lidar':[], 'recon':[], 'snodas': []}
        self.est_std_dict = {'kNN':[], 'kNN_GP':[], 'lidar':[], 'recon':[], 'snodas': []}
        self.est_stats = {'R2':{'kNN':[], 'kNN_GP':[], 'recon':[], 'snodas': []},
                          'slope':{'kNN':[], 'kNN_GP':[], 'recon':[], 'snodas': []},
                          'intercept':{'kNN':[], 'kNN_GP':[], 'recon':[], 'snodas': []},
                          'RMSE':{'kNN':[], 'kNN_GP':[], 'recon':[], 'snodas': []},
                          'MAE':{'kNN':[], 'kNN_GP':[], 'recon':[], 'snodas': []}}
        self.est_vs_gt_kde = {'kNN':[], 'kNN_GP':[], 'recon':[], 'snodas':[]}
        self.est_sensor = []
        self.est_residual = []
        self.est_raw_dict = {'kNN':[], 'kNN_GP':[], 'lidar':[], 'recon':[], 'snodas': [], 'elev':[]}
    
    # Construct kNN features for particular year and basin, only needs to run once
    def kNN_feature_construct(self, sensor_loc):
        recon_dir = site_name_abbr[self.site_name].upper() + "_recon/"
        year_range = range(2001, 2014)
        historical_fn_list = []
        recon_ts = None
        for year in year_range:
            temp_date = date(year, 4, 1)
            ending_date = date(year, 8, 31)
            while temp_date <= ending_date:
                temp_fn = recon_dir + temp_date.strftime("%d%b%Y").upper() + ".tif"
                historical_fn_list.append(temp_fn)
                temp_recon = gdal.Open(temp_fn).ReadAsArray()
                temp_date += timedelta(days=1)
                if recon_ts is None:
                    recon_ts = temp_recon[sensor_loc]
                else:
                    recon_ts = np.vstack((recon_ts, temp_recon[sensor_loc]))
        dump_fn = "kNN_training_testing/library_" + site_name_abbr[self.site_name].lower() + "_2001_2013_filenames.p"
        pickle.dump(historical_fn_list, open(dump_fn, 'wb'))
        np.save("kNN_training_testing/library_" + site_name_abbr[self.site_name].lower() + "_2001_2013.npy", recon_ts)

        sensor_data = None
        for temp_date in self.date_list:
            lidar_fn = "ASO_Lidar/" + site_name_abbr[self.site_name].upper() + temp_date.strftime("%Y%m%d") + "_500m.tif"
            lidar_swe = gdal.Open(lidar_fn).ReadAsArray() * density_factor[self.site_name]
            if sensor_data is None:
                sensor_data = lidar_swe[sensor_loc]
            else:
                sensor_data = np.vstack((sensor_data, lidar_swe[sensor_loc]))
        sensor_data[sensor_data < 0] = 0.
        sensor_data[np.isnan(sensor_data)] = 0.
        np.save("kNN_training_testing/" + site_name_abbr[self.site_name] + "_" +str(self.year) + \
                "_aso_simulated_sensor_data.npy", sensor_data)
    
    # load k nearest neighbor training and testing data into memory
    def load_kNN_data(self):
        self.recon_ts = np.load("kNN_training_testing/library_" + site_name_abbr[self.site_name]+ "_2001_2013.npy")
        self.recon_fn = pickle.load(open("kNN_training_testing/library_" + site_name_abbr[self.site_name] + \
                                           "_2001_2013_filenames.p", "rb"))
        self.sensor = np.load("kNN_training_testing/" + site_name_abbr[self.site_name] + "_" +str(self.year) + \
                              "_aso_simulated_sensor_data.npy")
        emp_cov = EmpiricalCovariance().fit(self.recon_ts)
        emp_cov_matrix = emp_cov.get_precision()
        dist = DistanceMetric.get_metric('mahalanobis', V=emp_cov_matrix)
        self.kNN = BallTree(self.recon_ts, metric=dist)
    
    # estimate the SWE for the 2D case for all days
    def kNN_predict(self):
        self.load_kNN_data()
        map(self.kNN_predict_mapper, zip(self.sensor, self.date_list))
       
    # Estimate SWE for each individual day
    def kNN_predict_mapper(self, sensor_date_tuple):
        sensor = sensor_date_tuple[0]
        temp_date = sensor_date_tuple[1]
        dem = gdal.Open("ASO_Lidar/" + self.site_name + "_500m_DEM.tif").ReadAsArray()
        dist, idx = self.kNN.query(np.array([sensor]), k=self.k)
        temp_fn_list = [self.recon_fn[i] for i in idx[0]]
        self.est_sensor.append(np.nanmean(self.recon_ts[idx[0]], axis=0))
        self.est_residual.append(sensor - self.est_sensor[-1])

        # Compute the sum of all k-nearest neighbors
        kNN_map_sum = 0.
        for temp_fn in temp_fn_list:
            kNN_map_sum += gdal.Open(temp_fn).ReadAsArray()

        # Compute the avg of the k-nearest neighbors
        kNN_map_avg = kNN_map_sum / float(self.k)

        # Load lidar, reconstruction, snodas at the same date
        lidar_map = gdal.Open("ASO_Lidar/"+site_name_abbr[self.site_name].upper() + \
                              temp_date.strftime("%Y%m%d") + "_500m.tif").ReadAsArray() * density_factor[self.site_name]

        if self.year <= 2014:
            recon_map = gdal.Open(site_name_abbr[self.site_name].upper() + \
                                  "_recon/"+temp_date.strftime("%d%b%Y").upper()+".tif").ReadAsArray()

        snodas_map = gdal.Open("SNODAS/" + site_name_abbr[self.site_name].upper() + "_" + \
                               temp_date.strftime("%Y%m%d") + ".tif").ReadAsArray() / 1000.

        # Filter these data by lidar value and store them in the est_dict
        self.est_dict['kNN'].append(kNN_map_avg[lidar_map >= 0.])
        self.est_dict['snodas'].append(snodas_map[lidar_map >= 0.])
        self.est_dict['lidar'].append(lidar_map[lidar_map >= 0.])
        self.est_dict['elev'].append(dem[lidar_map >= 0.])
        self.est_dict['kNN_GP'].append(0)
        if self.year <= 2014:
            self.est_dict['recon'].append(recon_map[lidar_map >= 0.])

        # Do not filter these data and store them in the est_raw_dict
        self.est_raw_dict['kNN'].append(kNN_map_avg)
        self.est_raw_dict['snodas'].append(snodas_map)
        self.est_raw_dict['lidar'].append(lidar_map)
        self.est_raw_dict['elev'].append(dem)
        self.est_raw_dict['kNN_GP'].append(0)
        if self.year <= 2014:
            self.est_raw_dict['recon'].append(recon_map)

    def _kNN_predict_custom_k_rmse(self, k, sensor, temp_date):
        dist, idx = self.kNN.query(np.array([sensor]), k=k)
        temp_fn_list = [self.recon_fn[i] for i in idx[0]]
        kNN_map_sum = 0.
        for temp_fn in temp_fn_list:
            kNN_map_sum += gdal.Open(temp_fn).ReadAsArray()
        kNN_map_avg = kNN_map_sum / float(k)
        lidar_map = gdal.Open("ASO_Lidar/" + site_name_abbr[self.site_name].upper() + \
                              temp_date.strftime("%Y%m%d").upper()+".tif").ReadAsArray()
        kNN_map_avg = kNN_map_avg[lidar_map>=0]
        lidar_map = lidar_map[lidar_map>=0]
        return np.sqrt(mse(kNN_map_avg, lidar_map))

    def kNN_update_est_stats(self):
        for key_1 in self.est_stats:
            for key_2 in self.est_stats[key_1]:
                self.est_stats[key_1][key_2] = []
        for p in self.products:
            for idx in range(len(self.date_list)):
                slope, intercept, r_value, p_value, std_err = linregress(self.est_dict[self.groundTruth][idx], self.est_dict[p][idx])
                self.est_stats['R2'][p].append(r_value**2)
                self.est_stats['slope'][p].append(slope)
                self.est_stats['intercept'][p].append(intercept)
                self.est_stats['RMSE'][p].append(np.sqrt(mse(self.est_dict[self.groundTruth][idx], self.est_dict[p][idx])))
                self.est_stats['MAE'][p].append(mae(self.est_dict[self.groundTruth][idx], self.est_dict[p][idx]))

    # Compute the paired kernel density estimation for a single data
    def _compute_pair_kde_single(self, product, idx):
        x_1 = self.est_dict[self.groundTruth][idx]
        x_2 = self.est_dict[product][idx]
        bivariate_x = np.column_stack(((x_1, x_2)))
        kde = gaussian_kde(bivariate_x)
        self.est_vs_gt_kde[product][idx] = kde(bivariate_x)

    def compute_pair_kde(self):
        # initialize kde
        for key in self.est_vs_gt_kde:
            self.est_vs_gt_kde[key] = [0] * self.num_days

        # iterate through all the products
        for product in self.products:
            # iterate through all the days
            for idx in range(self.num_days):
                t = threading.Thread(target=self._compute_pair_kde_single, args=(product, idx))
                t.start()

    # plot model results vs lidar
    def kNN_recon_snodas_vs_lidar(self):
        if self.year <= 2014:
            recon_exist=True

        self.compute_pair_kde()

        fig, axarr = plt.subplots(ncols = len(self.products), nrows = len(self.date_list),
                                  figsize=(2.5 * len(self.products), 2.5 * len(self.date_list)))

        # Iterate through all possible dates
        for j, temp_date in enumerate(self.date_list):

            # Iterate through all the products and make scatter plots with regression line
            for i, p in enumerate(self.products):
                axarr[j, i].scatter(self.est_dict[self.groundTruth][j], self.est_dict[p][j], c=self.est_vs_gt_kde[p][j],
                                    s=2, edgecolor='none', cmap='cool', alpha=0.5)
                axarr[j, i].plot([0, 1], [0, 1], '--k')
                axarr[j, i].plot([0, 1], [self.est_stats['intercept'][p][j],
                                          self.est_stats['slope'][p][j] + self.est_stats['intercept'][p][j]], '-b')
                if self.year == 2014:
                    axarr[j, i].set_xlim([0, 1])
                    axarr[j, i].set_ylim([0, 1])
                else:
                    axarr[j, i].set_xlim([0, 1.4])
                    axarr[j, i].set_ylim([0, 1.4])

            # Iterate through all possible combination of products and time
            if j < len(self.date_list)-1:
                for i in range(len(self.products)):
                    axarr[j, i].xaxis.set_ticklabels([])
                if self.year == 2014:
                    axarr[j, 0].yaxis.set_ticklabels(["", "0.2", "0.4", "0.6", "0.8", "1.0"])
                else:
                    axarr[j, 0].yaxis.set_ticklabels(["", "0.2", "0.4", "0.6", "0.8", "1.0", "1.2", "1.4"])
            
            # Set all the yaxis from 2nd column to nothing
            for i in range(1, len(self.products)):
                axarr[j, i].yaxis.set_ticklabels([])

            # Set all xaxis and yaxis on the lowest row
            if self.year == 2014:
                if j==len(self.date_list)-1:
                    for i in range(1, len(self.products)):
                        axarr[j, i].xaxis.set_ticklabels(["", "0.2", "0.4", "0.6", "0.8", "1.0"])
                    axarr[j, 0].xaxis.set_ticklabels(["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"])
                    axarr[j, 0].yaxis.set_ticklabels(["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"])
            else:
                if j==len(self.date_list)-1:
                    for i in range(1, len(self.products)):
                        axarr[j, i].xaxis.set_ticklabels(["", "0.2", "0.4", "0.6", "0.8", "1.0", "1.2", "1.4"])
                    axarr[j, 0].xaxis.set_ticklabels(["0.0", "0.2", "0.4", "0.6", "0.8", "1.0", "1.2", "1.4"])
                    axarr[j, 0].yaxis.set_ticklabels(["0.0", "0.2", "0.4", "0.6", "0.8", "1.0", "1.2", "1.4"])

            axarr[j, len(self.products)-1].text(0.55, 0.1, temp_date.strftime("%Y-%m-%d"), fontsize=8)

        fig.text(0.35, 0.09, 'Lidar measured SWE, m')
        fig.text(0.03, 0.6, 'kNN interpolated SWE, m', rotation=90)
        fig.text(0.91, 0.6, self.products[-1] + ' SWE, m', rotation=270)
        plt.subplots_adjust(left=0.14, wspace=0, hspace=0)
        figFn = self.site_name + "_simulated_"
        for p in self.products:
            figFn += p + "_"
        figFn += "vs_lidar_" + str(self.year) + ".pdf"
        plt.savefig(figFn, bbox_inches='tight')
        plt.show()

    def tune_k(self):
        np.random.seed(1)
        # self.recon_ts, self.recon_fn, self.sensor are ready
        self.k_list = range(50)
        # k_rmse_dict would be (key: date, value: rmse list at different k value)
        self.k_rmse_dict = {}
        for sensor, temp_date in zip(self.sensor, self.date_list):
            temp_rmse = []
            for k in self.k_list:
                temp_rmse.append(self._kNN_predict_custom_k_rmse(k, sensor, temp_date))
            self.k_rmse_dict[temp_date] = temp_rmse

    def error_vs_num_scenes(self):
        pass
    
    # Show scatterplot statistics of the
    def scatter_statistics_figure(self):
        fig, axarr = plt.subplots(ncols=1, nrows=2, figsize=(3.5, 4), sharex=True)
        axarr[0].plot(self.date_list, self.est_stats['slope']['kNN'], '-k')
        axarr[1].plot(self.date_list, self.est_stats['R2']['kNN'], '-k')

        # If earlier than 2014, plot reconstruction data as well
        if self.year <= 2014:
            axarr[0].plot(self.date_list, self.est_stats['slope']['recon'], '--k')
            axarr[1].plot(self.date_list, self.est_stats['R2']['recon'], '--k')
            axarr[0].legend(["kNN vs. Lidar", "Reconstruction vs. Lidar"], fontsize=10, frameon=False, loc=4)
        else:
            axarr[0].legend(['kNN vs. Lidar'], fontsize=10, frameon=False, loc=4)

        # Configure the axis and labels
        months = MonthLocator(range(1, 13), bymonthday=1, interval=3)
        monthsFmt = DateFormatter("%b %y")
        axarr[0].set_ylabel("Slopes")
        if self.site_name == 'Merced':
            axarr[0].set_ylim([0.6, 1.6])
            axarr[0].yaxis.set_ticks([0.6, 0.8, 1., 1.2, 1.4, 1.6])
            axarr[1].yaxis.set_ticks([0.55, 0.65, 0.75, 0.85])
        else:
            axarr[0].set_ylim([min(self.est_stats['slope']['kNN'])-0.1, max(self.est_stats['slope']['kNN'])+0.1])
        axarr[1].xaxis.set_major_locator(months)
        axarr[1].set_ylabel("Correlation coefficients")
        axarr[1].xaxis.set_major_formatter(monthsFmt)
        plt.savefig(self.site_name + "_slope_r_kNN_recon_Lidar_" + str(self.year) + ".pdf", bbox_inches='tight')
        plt.show()
    
    def kNN_update_mean_std(self):
        # Clear what we currently have
        for key in self.est_mean_dict:
            self.est_mean_dict[key] = []
            self.est_std_dict[key] = []

        # Populating the mean and std into the dict
        for j, temp_date in enumerate(self.date_list):
            for p in (self.products + [self.groundTruth]):
                self.est_mean_dict[p].append(np.nanmean(self.est_dict[p][j]))
                self.est_std_dict[p].append(np.nanstd(self.est_dict[p][j]))

    def kNN_mean_std_ts(self, snodas=False):
        colors = ['r', 'g', 'b', 'k']
        linestyles = ['-', '--', '-.', ':']
        plt_lines = []
        plt_lines_title = []
        mean_std_max = 0.
        for p in (self.products + [self.groundTruth]):
            if p != "snodas":
                tempLine = plt.errorbar(self.date_list, self.est_mean_dict[p], self.est_std_dict[p], linestyle=linestyles.pop(0), marker='o', color=colors.pop(0))
                plt_lines.append(tempLine)
                plt_lines_title.append(p)
                mean_std_max = max([x+y for x, y in zip(self.est_mean_dict[p], self.est_std_dict[p])] + [mean_std_max])
            else:
                if snodas:
                    tempLine = plt.errorbar(self.date_list, self.est_mean_dict[p], self.est_std_dict[p], linestyle=linestyles.pop(0), marker='o', color=colors.pop(0))
                    plt_lines.append(tempLine)
                    plt_lines_title.append(p)
                    mean_std_max = max([x+y for x, y in zip(self.est_mean_dict[p], self.est_std_dict[p])] + [mean_std_max])

        plt.xlim([min(self.date_list) - timedelta(days=5), max(self.date_list) + timedelta(days=5)])
        plt.grid()
        plt.ylim([0, mean_std_max * 1.1])
        
        # Define date_locators separate x-axis by 14 days
        date_locators = [min(self.date_list) + timedelta(days=dt) 
                         for dt in range(0, (max(self.date_list) - min(self.date_list)).days + 5, 14)]
        
        date_locators_string = [temp_date.strftime('%b %d %Y') for temp_date in date_locators]
        plt.legend(plt_lines, plt_lines_title, numpoints=1)
        plt.xlabel('Date')
        plt.ylabel('SWE, m')
        plt.xticks(date_locators, date_locators_string)
        plt.savefig(self.site_name.upper() + "_basin_wide_mean_std_" + str(self.year) + ".pdf", dpi=100)
        plt.show()
        
    def elev_band_mean_std_comparison(self, snodas=False):

        def elevation_avg(idx):
            # Generate an elevation gradient
            elevation_gradient = np.linspace(1500, np.max(self.est_dict['elev'][idx]), 40)

            # Generate a numpy array with enough space for elevation and avg and std for all different kind of products
            prod_avg_std = np.zeros((len(elevation_gradient)-1, (len(self.products)+1)*2+1))

            # Iterate through all elevation bands
            for i, temp_elev in enumerate(elevation_gradient[:-1]):

                # Calculate the min and max of this elevation band
                min_elev = temp_elev
                max_elev = elevation_gradient[i+1]
                avg_elev = (min_elev + max_elev)/2.

                # Initialize the mean_std list with elevation of this band
                temp_prod_avg_std = [avg_elev]

                # Iterate through all different products and calculate their mean and standard deviation
                for p in (self.products + [self.groundTruth]):
                    temp_prod_avg_std.append(np.nanmean(self.est_dict[p][idx][np.logical_and(self.est_dict['elev'][idx]>=min_elev,
                        self.est_dict['elev'][idx]<=max_elev)]))
                    temp_prod_avg_std.append(np.nanstd(self.est_dict[p][idx][np.logical_and(self.est_dict['elev'][idx]>=min_elev,
                        self.est_dict['elev'][idx]<=max_elev)]))

                # Assign the temporary mean_std to the entire numpy array
                prod_avg_std[i] = temp_prod_avg_std

            return prod_avg_std

        # Initialize the figure
        fig, axarr = plt.subplots(ncols=2, nrows=len(self.date_list), figsize=(5, 10))

        # calculate the mean_std of swe of each elevation band
        swe_avg_by_elev_list = map(elevation_avg, range(len(self.date_list)))

        # iterate through all dates in this year
        for i, temp_date in enumerate(self.date_list):
            swe_avg_by_elev = swe_avg_by_elev_list[i]

            colors = ['-r', '-g', '-b', '-y']
            facecolors = ['red', 'green', 'blue', 'yellow']
            plt_lines_title = []
            plt_lines = []
            
            # plot the mean and std at each elevation band
            for j, p in enumerate(self.products + [self.groundTruth]):
                if p != 'snodas':
                    tempLine, = axarr[i, 0].plot(swe_avg_by_elev[:, 0], swe_avg_by_elev[:, j*2+1], colors.pop(0))
                    axarr[i, 1].fill_between(swe_avg_by_elev[:, 0], 0, swe_avg_by_elev[:, j*2+2], facecolor=facecolors.pop(0), alpha=0.3)
                    plt_lines.append(tempLine)
                    plt_lines_title.append(p)
                else:
                    if snodas:
                        tempLine, = axarr[i, 0].plot(swe_avg_by_elev[:, 0], swe_avg_by_elev[:, j*2+1], colors.pop(0))
                        axarr[i, 1].fill_between(swe_avg_by_elev[:, 0], 0, swe_avg_by_elev[:, j*2+2], facecolor=facecolors.pop(0), alpha=0.3)
                        plt_lines.append(tempLine)
                        plt_lines_title.append(p)

            if i < len(self.date_list) - 1:
                axarr[i, 0].xaxis.set_ticklabels([])
                axarr[i, 1].xaxis.set_ticklabels([])
            if self.year==2014:
                axarr[i, 0].yaxis.set_ticks([0, 0.2, 0.4, 0.6, 0.8])
                axarr[i, 0].grid()
                axarr[i, 1].set_ylim([0.0, 0.3])
                axarr[i, 1].yaxis.set_ticks([0.0, 0.1, 0.2, 0.3])
            else:
                axarr[i, 0].yaxis.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
                axarr[i, 0].grid()
                axarr[i, 1].set_ylim([0.0, 0.5])
                axarr[i, 1].yaxis.set_ticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
            axarr[i, 1].grid()
        axarr[len(self.date_list)-1, 0].set_xlabel('Elevation, m')
        axarr[len(self.date_list)-1, 1].set_xlabel('Elevation, m')
        
        # setup legends
        axarr[0, 0].legend(plt_lines, 
                plt_lines_title, 
                loc=2, 
                frameon=False, 
                fontsize=10)

        fig.text(0.03, 0.57, 'Mean SWE, m', rotation=90)
        fig.text(0.93, 0.6, 'Standard deviation of SWE, m', rotation=270)
        plt.savefig(self.site_name + '_swe_elevation_mean_std_' + str(self.year) + '.pdf', dpi=100)
        plt.show()