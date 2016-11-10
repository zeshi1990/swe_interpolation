import gdal
from datetime import timedelta
from scipy.stats import pearsonr, linregress
from sklearn.covariance import EmpiricalCovariance
from sklearn.neighbors import BallTree, DistanceMetric
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from matplotlib import pyplot as plt

site_name_abbr = {"Merced": "mb", "Tuolumne":"tb"}
density_factor = {"Merced": 0.333, "Tuolumne": 1.0}

class kNN_swe_regressor():
    def __init__(self, site_name, year, date_list, k):
        self.site_name = site_name
        self.year = year
        self.date_list = date_list[self.year][site_name]
        self.k = k
        self.kNN = None
        self.groundTruth = 'lidar'
        self.products = ['kNN', 'snodas']
        if self.year <= 2014:
            self.model.append('recon')
        self.est_dict = {'kNN':[], 'lidar':[], 'recon':[], 'snodas': [], 'elev':[]}
        self.est_mean_dict = {'kNN':[], 'lidar':[], 'recon':[], 'snodas': []}
        self.est_std_dict = {'kNN':[], 'lidar':[], 'recon':[], 'snodas': []}
        self.est_stats = {'R2':{'kNN':[], 'recon':[], 'snodas': []}, 
                          'slope':{'kNN':[], 'recon':[], 'snodas': []},
                          'intercept':{'kNN':[], 'recon':[], 'snodas': []},
                          'RMSE':{'kNN':[], 'recon':[], 'snodas': []},
                          'MAE':{'kNN':[], 'recon':[], 'snodas': []}}
    
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
            lidar_fn = "ASO_Lidar/" + site_name_abbr[self.site_name] + temp_date.strftime("%Y%m%d") + "_500m.tif"
            lidar_swe = gdal.Open(lidar_fn).ReadAsArray() * density_factor[self.site_name]
            if sensor_data is None:
                sensor_data = lidar_swe[sensor_loc]
            else:
                sensor_data = np.vstack((sensor_data, lidar_swe[sensor_loc]))
        np.save("kNN_training_testing/" + site_name_abbr[self.site_name] + "_" +str(self.year) + \
                "_aso_simulated_sensor_data.npy", sensor_data)
    
    # load k nearest neighbor training and testing data into memory
    def load_kNN_data(self):
        self.recon_ts = np.load("kNN_training_testing/library_" + site_name_abbr[self.site_name].lower() + "_2001_2013.npy")
        self.recon_fn = pickle.pickle.load(open("kNN_training_testing/library_" + site_name_abbr[self.site_name].lower() + \
                                           "mb_2001_2013_filenames.p", "rb"))
        self.sensor = np.load("kNN_training_testing/" + site_name_abbr[self.site_name] + "_" +str(self.year) + \
                              "_aso_simulated_sensor_data.npy")
        emp_cov = EmpiricalCovariance().fit(self.recon_ts)
        emp_cov_matrix = emp_cov.get_precision()
        dist = DistanceMetric.get_metric('mahalanobis', V=emp_cov_matrix)
        self.kNN = BallTree(recon_ts, metric=dist)
        self.sensor = lidar_sensor_data
    
    # estimate the SWE for the 2D case for all days
    def kNN_predict(self):
        self.load_kNN_data()
        map(self.kNN_predict_mapper, zip(self.sensor, self.date_list))
       
    # Estimate SWE for each individual day
    def kNN_predict_mapper(self, sensor_date_tuple):
        sensor = sensor_date_tuple[0]
        temp_date = sensor_date_tupe[1]
        dem = gdal.Open("ASO_Lidar/" + self.site_name + "_DEM_500m.tif").ReadAsArray()
        dist, idx = self.kNN.query(np.array([sensor]), k=self.k)
        temp_fn_list = [self.recon_fn[i] for i in idx[0]]

        # Compute the sum of all k-nearest neighbors
        kNN_map_sum = 0.
        for temp_fn in temp_fn_list:
            kNN_map_sum += gdal.Open(temp_fn).ReadAsArray()

        # Compute the avg of the k-nearest neighbors
        kNN_map_avg = kNN_map_sum / float(k)

        # Load lidar, reconstruction, snodas at the same date
        lidar_map = gdal.Open("ASO_Lidar/"+site_name_abbr[self.site_name].upper() + \
                              temp_date.strftime("%Y%m%d") + "_500m.tif").ReadAsArray() * density_factor[self.site_name]

        if self.year <= 2014:
            recon_map = gdal.Open(site_name_abbr[self.site_name].upper() + \
                                  "_recon/"+temp_date.strftime("%d%b%Y").upper()+".tif").ReadAsArray()

        snodas_map = gdal.Open("SNODAS/" + site_name_abbr[self.site_name].upper() + "_" + \
                               temp_date.strftime("%Y%m%d") + ".tif").ReadAsArray() / 1000.

        # Filter these data by lidar value and store them in the est_dictionary
        self.est_dict['kNN'].append(kNN_map_avg[lidar_map >= 0.])
        self.est_dict['snodas'].append(snodas_map[lidar_map >= 0.])
        self.est_dict['lidar'].append(lidar_map[lidar_map >= 0.])
        self.est_dict['elev'].append(dem[lidar_map >= 0.])
        if self.year <= 2014:
            self.est_dict['recon'].append(recon_map[lidar_map >= 0.])

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

    # plot model results vs lidar
    def kNN_recon_snodas_vs_lidar(self):
        if self.year <= 2014:
            recon_exist=True

        fig, axarr = plt.subplots(ncols = len(self.products), nrows = len(self.date_list), figsize=(2.5 * len(self.products), 2.5 * len(self.date_list)))

        # Iterate through all possible dates
        for j, temp_date in enumerate(self.date_list):

            # Iterate through all the products and make scatter plots with regression line
            for i, p in enumerate(self.products):
                axarr[j, i].plot(self.est_dict[self.groundTruth][j], self.est_dict[p][j], '.k', markersize=2)
                axarr[j, i].plot([0, 1], [0, 1], '--k')
                axarr[j, i].plot(self.est_stats['intercept'][p], self.est_stats['slope'][p] + self.est_stats['intercept'][p], '-b')
                axarr[j, i].set_xlim([0, 1])
                axarr[j, i].set_ylim([0, 1])

            # Iterate through all possible combination of products and time
            if j < len(self.date_list)-1:
                for i in range(len(self.products)):
                    axarr[j, i].xaxis.set_ticklabels([])
                axarr[j, 0].yaxis.set_ticklabels(["", "0.2", "0.4", "0.6", "0.8", "1.0"])
            
            # Set all the yaxis from 2nd column to nothing
            for i in range(1, len(self.products)):
                axarr[j, i].yaxis.set_ticklabels([])

            # Set all xaxis and yaxis on the lowest row
            if j==len(self.date_list)-1:
                for i in range(1, len(self.products)):
                    axarr[j, i].xaxis.set_ticklabels(["", "0.2", "0.4", "0.6", "0.8", "1.0"])
                axarr[j, 0].xaxis.set_ticklabels(["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"])
                axarr[j, 0].yaxis.set_ticklabels(["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"])

            axarr[j, len(self.products)-1].text(0.55, 0.1, temp_date.strftime("%Y-%m-%d"), fontsize=8)

        fig.text(0.35, 0.09, 'Lidar measured SWE, m')
        fig.text(0.03, 0.6, 'kNN interpolated SWE, m', rotation=90)
        fig.text(0.91, 0.6, 'SNODAS SWE, m', rotation=270)
        plt.subplots_adjust(left=0.14, wspace=0, hspace=0)
        figFn = self.site_name + "_simulated_"
        for p in self.products:
            figFn += p + "_"
        figFn += "vs_lidar.pdf"
        plt.savefig(figFn, bbox_inches='tight')
        plt.show()
    
    # Show scatterplot statistics of the
    def scatter_statistics_figure(self):
        fig, axarr = plt.subplots(ncols=1, nrows=2, figsize=(3.5, 4), sharex=True)
        axarr[0].plot(self.date_list, self.est_stats['slope']['kNN'], '-k')
        axarr[1].plot(date_list, self.est_stats['R2']['kNN'], '-k')

        # If earlier than 2014, plot reconstruction data as well
        if self.year <= 2014:
            axarr[0].plot(self.date_list, self.est_stats['slope']['recon'], '--k')
            axarr[1].plot(date_list, self.est_stats['R2']['recon'], '--k')
            axarr[0].legend(["kNN vs. Lidar", "Reconstruction vs. Lidar"], fontsize=10, frameon=False, loc=4)
        else:
            axarr[0].legend(['kNN vs. Lidar'], fontsize=10, frameon=False, loc=4)

        # Configure the axis and labels
        axarr[0].set_ylabel("Slopes")
        axarr[0].set_ylim([0.6, 1.6])
        axarr[0].yaxis.set_ticks([0.6, 0.8, 1., 1.2, 1.4, 1.6])
        axarr[1].xaxis.set_major_locator(months)
        axarr[1].yaxis.set_ticks([0.55, 0.65, 0.75, 0.85])
        axarr[1].set_ylabel("Correlation coefficients")
        axarr[1].xaxis.set_major_formatter(monthsFmt)
        plt.savefig(self.site_name + "_slope_r_kNN_recon_Lidar.pdf", bbox_inches='tight')
        plt.show()
    
    def kNN_update_mean_std(self):
        # Clear what we currently have
        for key in self.est_mean_dict:
            self.est_mean_dict[key] = []
            self.est_std_dict[key] = []

        # Populating the mean and std into the dict
        for j, temp_date in enumerate(date_list):
            for p in (self.products + [self.groundTruth]):
                self.est_mean_dict[p].append(np.nanmean(self.est_dict[p][j]))
                self.est_std_dict[p].append(np.nanstd(self.est_dict[p][j]))

    def kNN_mean_std_ts(self, snodas=False):
        colors = ['r', 'g', 'b', 'k']
        linestyles = ['-', '--', '-.', ':']
        plt_lines = []
        plt_lines_title = []
        for p in (self.products + [self.groundTruth]):
            tempLine = plt.errorbar(self.date_list, self.est_mean_dict[p], self.est_std_dict[p], linestyle=linestyles.pop(0), marker='o', color=colors.pop(0))
            plt_lines.append(tempLine)
            plt_lines_title.append(p)

        plt.xlim([min(self.date_list) - timedelta(days=5), max(self.date_list) + timedelta(days=5)])
        plt.grid()
        plt.ylim([0, max(max(kNN_mean_list), max(lidar_mean_list), max(recon_mean_list), max(snodas_mean_list)) + \
                  max(max(kNN_std_list), max(lidar_std_list), max(recon_std_list), max(snodas_std_list))])
        
        # Define date_locators separate x-axis by 14 days
        date_locators = [min(self.date_list) + timedelta(days=dt) 
                         for dt in range(0, (max(self.date_list) - min(self.date_list)).days + 5, 14)]
        
        date_locators_string = [temp_date.strftime('%b %d %Y') for temp_date in date_locators]
        plt.legend([kNN_mark, lidar_mark, recon_mark], ['kNN', 'Lidar', 'Reconstruction'], numpoints=1)
        plt.xlabel('Date')
        plt.ylabel('SWE, m')
        plt.xticks(date_locators, date_locators_string)
        plt.savefig(self.site_name.upper() + "_basin_wide_mean_std.pdf", dpi=100)
        plt.show()
        
    def elev_band_mean_std_comparison(self, snodas=False):

        def elevation_avg(est_tuple):
            elevation_gradient = np.linspace(1500, np.max(est_tuple[-1]), 40)
            new_features = np.zeros((len(elevation_gradient)-1, 9))

            for i, temp_elev in enumerate(elevation_gradient[:-1]):
                min_elev = temp_elev
                max_elev = elevation_gradient[i+1]
                avg_elev = (min_elev + max_elev)/2.

                # Index the data that within the elevation range
                temp_kNN = est_tuple[0][np.logical_and(est_tuple[-1] >= min_elev, est_tuple[-1] <= max_elev)]
                temp_lidar = est_tuple[3][np.logical_and(est_tuple[-1] >= min_elev, est_tuple[-1] <= max_elev)]
                temp_recon = est_tuple[1][np.logical_and(est_tuple[-1] >= min_elev, est_tuple[-1] <= max_elev)]
                temp_snodas = est_tuple[2][np.logical_and(est_tuple[-1] >= min_elev, est_tuple[-1] <= max_elev)]

                # calculate the mean and std for each item(kNN, lidar, recon, snodas)
                kNN_mean = np.nanmean(temp_kNN)
                kNN_std = np.nanstd(temp_kNN)
                lidar_mean = np.nanmean(temp_lidar)
                lidar_std = np.nanstd(temp_lidar)
                recon_mean = np.nanmean(temp_recon)
                recon_std = np.nanstd(temp_recon)
                snodas_mean = np.nanmean(temp_snodas)
                snodas_std = np.nanmean(temp_snodas)
                new_features[i] = [avg_elev, kNN_mean, kNN_std, lidar_mean, lidar_std, recon_mean, recon_std, snodas_mean, snodas_std]

            return new_features

        fig, axarr = plt.subplots(ncols=2, nrows=len(date_list), figsize=(5, 10))
        swe_avg_by_elev_list = map(elevation_avg, self.est_tuple_list)
        for i, temp_date in enumerate(self.date_list):
            swe_avg_by_elev = swe_avg_by_elev[i]

            # plot the mean at each elevation band
            knn_line, = axarr[i, 0].plot(swe_avg_by_elev[:, 0], swe_avg_by_elev[:, 1], '-r')
            lidar_line, = axarr[i, 0].plot(swe_avg_by_elev[:, 0], swe_avg_by_elev[:, 3], '-g')
            recon_line, = axarr[i, 0].plot(swe_avg_by_elev[:, 0], swe_avg_by_elev[:, 5], '-b')

            if snodas:
                snodas_line, = axarr[i, 0].plot(swe_avg_by_elev[:, 0], swe_avg_by_elev[:, 7], '-y')

            # plot the std at each elevation band
            axarr[i, 1].fill_between(avg_features[:, 0], 0, avg_features[:, 2], facecolor='red')
            axarr[i, 1].fill_between(avg_features[:, 0], 0, avg_features[:, 4], facecolor='green', alpha=0.3)
            axarr[i, 1].fill_between(avg_features[:, 0], 0, avg_features[:, 6], facecolor='blue', alpha=0.3)

            if snodas:
                axarr[i, 1].fill_between(avg_features[:, 0], 0, avg_features[:, 8], facecolor='yellow', alpha=0.3)
            if i < len(self.date_list) - 1:
                axarr[i, 0].xaxis.set_ticklabels([])
                axarr[i, 1].xaxis.set_ticklabels([])
            axarr[i, 0].yaxis.set_ticks([0, 0.2, 0.4, 0.6, 0.8])
            axarr[i, 0].grid()
            axarr[i, 1].set_ylim([0.0, 0.3])
            axarr[i, 1].yaxis.set_ticks([0.0, 0.1, 0.2, 0.3])
            axarr[i, 1].grid()
        axarr[len(self.date_list)-1, 0].set_xlabel('Elevation, m')
        axarr[len(self.date_list)-1, 1].set_xlabel('Elevation, m')
        # setup legends
        if snodas:
            axarr[0, 0].legend([knn_line, lidar_line, recon_line, snodas_line], 
                ['kNN', 'Lidar', 'Reconstruction', 'SNODAS'], 
                loc=2, 
                frameon=False, 
                fontsize=10)
        else:
            axarr[0, 0].legend([knn_line, lidar_line, recon_line], 
                ['kNN', 'Lidar', 'Reconstruction'], 
                loc=2, 
                frameon=False, 
                fontsize=10)

        fig.text(0.03, 0.57, 'Mean SWE, m', rotation=90)
        fig.text(0.93, 0.6, 'Standard deviation of SWE, m', rotation=270)
        plt.savefig(self.site_name + '_swe_elevation_mean_std.pdf', dpi=100)
        plt.show()

