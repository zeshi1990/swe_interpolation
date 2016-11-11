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