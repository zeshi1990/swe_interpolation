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

		# setup the covariance functions with default settings
		self.update_cov_fcns()

		self.trainX = self.db.load_features(self.site_name, sensor_feature=True, exclude_null=True)
		self.testX = self.db.load_features(self.site_name, sensor_feature=False, exclude_null=False)


	def update_cov_fcns(self, rbf_ub=2000.0, matern_ub=4000.0, dotProduct_ub=1000.0):
		self.cov = sum([0.5 * RBF(length_scale=100.0, length_scale_bounds = (1e-1, rbf_ub)), 
					0.5 * Matern(length_scale=100.0, length_scale_bounds = (1e-1, matern_ub)),
					1.0 * DotProduct(sigma_0=100.0, sigma_0_bounds = (1e-2, dotProduct_ub)),
					1.0 * WhileKernel()])

	def gp_load_data(self, date_obj, estY=None):
		self.trainY = self.db.load_swe(date_obj, self.site_name, schema_name, sensor=True)
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