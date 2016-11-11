import numpy as np
import gdal
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern, DotProduct

class GP_swe_regressor:

	def __init__(self, db, site_name, residual):
		# self.db is the database connector to postgresql
		self.db = db

		# self.site_name is the site_name for this GP analysis
		self.site_name = site_name

		# boolean parameter, if residual is true, 
		# then the GP is based on the difference between the estimator
		self.residual = residual

		self.trainX = self.db.load_features(sensor_feature=True, exclude_null=True, basin=self.site_name)
		self.testX = self.db.load_features(sensor_feature=False, exclude_null=False, basin=self.site_name)



	def update_cov_fcns(self):
		self.cov = sum([0.5 * RBF(length_scale=100.0, length_scale_bounds = (1e-1, 2000.0)), 
					0.5 * Matern(length_scale=100.0, length_scale_bounds = (1e-1, 4000.0)),
					1.0 * DotProduct(sigma_0=100.0, sigma_0_bounds = (1e-2, 1000.0)),
					1.0 * WhileKernel()])

	def gp_load_data(self, date, estY=None):
		self.trainY = self.db.load_swe(date, schema_name, sensor=True, basin=self.site_name)
		if self.residual:
			self.trainY -= estY
		self.testY = None

	def gp_train(self):
		self.GP = GaussianProcessRegressor(kernel=self.cov, alpha=0.0)
		if len(self.trainX.shape) == 1:
			self.GP.fit(self.trainX[:, np.newaxis], self.trainY)
		else:
			self.GP.fit(self.trainX, self.trainY)

	def gp_predict(self):
		self.testY = self.GP.predict(self.testX)

	def update_kNN(self, kNN_regressor, idx):
		if self.testY is not None and self.residual:
			kNN_regressor.est_raw_dict['kNN'][idx] += self.testY.reshape(kNN_regressor.est_raw_dict['kNN'][idx].shape)


	# This function is used for loading the features


class dbTools:
	from .postgisUtil import postgisUtil
	def __init__(database, username):
		self.db = postgisUtil(database=database, username=username)

	def load_features(self, sensor_feature=False, exclude_null=True, basin='Merced'):
	    # Please note that the indices here are indices of python, starting from 0
	    features = ['DEM', 'SLP', 'ASP', 'VEG']
	    if not sensor_feature:
	        for i, feature in enumerate(features):
	            feature_array = self.db.query_map(feature, 'topo', basin.lower())
	            if i == 0:
	                grid_y, grid_x = np.meshgrid(range(feature_array.shape[0]), range(feature_array.shape[1]), indexing='ij')
	                grid_y_array, grid_x_array = grid_y.flatten(), grid_x.flatten()
	                feature_space = np.column_stack((grid_y_array, np.column_stack((grid_x_array, feature_array.flatten()))))
	            else:
	                feature_space = np.column_stack((feature_space, feature_array.flatten()))
	        if exclude_null:
	            feature_space = feature_space[feature_space[:, 2] >= 0]
	        feature_space[feature_space < 0] = np.nan
	    else:
	        feature_array = self.db.geoms_table_to_map_pixel_values(features, 'sensors', basin.lower(), 'site_coords', 'topo', basin.lower())
	        spatial_feature = self.db.geoms_table_to_map_pixel_indices(features[0], 'sensors', basin.lower(), 'site_coords', 'topo', basin.lower())
	        feature_space = np.column_stack((spatial_feature, feature_array))
	    return feature_space

	def load_swe(self, date_obj, schema_name, sensor=False, basin="Merced"):
		if type(date_obj)==str:
		    date_str = date_obj
		elif type(date_obj)==date:
		    date_str = date_obj.strftime("%Y%m%d")
		else:
		    print "The input date_obj dtype is not supported"
		    return
		if not sensor:
		    DEM = self.db.query_map("DEM", 'topo', basin.lower())
		    swe = self.db.query_map(date_str, schema_name, basin.lower())
		    data_array = np.column_stack((DEM.flatten(), swe.flatten()))
		    data_array = data_array[data_array[:, 0] >= 0]
		    data_array = data_array[:, 1]
		else:
		    data_array = self.db.geoms_table_to_map_pixel_values(date_str, 'sensors', basin.lower(), 'site_coords', schema_name, basin.lower())
		return data_array