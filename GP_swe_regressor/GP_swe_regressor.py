import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern, DotProduct


class GP_swe_regressor:

    def __init__(self, db, site_name, residual=None):
        # self.db is the database connector to postgresql
        self.db = db

        # self.site_name is the site_name for this GP analysis
        self.site_name = site_name

        # boolean parameter, if residual is true,
        # then the GP is based on the difference between the estimator
        self.residual = residual

        # setup the covariance functions with default settings
        self.update_cov_fcns()

        self.trainX = self.db.load_features(self.site_name, sensor=True, exclude_null=True)
        self.testX_all = self.db.load_features(self.site_name, sensor=False, exclude_null=False)

        self.testX_finite_bool = np.isfinite(self.testX_all[:, 2])
        self.testX = self.testX_all[self.testX_finite_bool]
        self.testY = None

    def update_residual(self, residual):
        self.residual = residual


    def update_cov_fcns(self, rbf_ub=2000.0, matern_ub=4000.0, dotProduct_ub=1000.0):
        self.cov = sum([0.5 * RBF(length_scale=100.0, length_scale_bounds = (1e-1, rbf_ub)),
                    0.5 * Matern(length_scale=100.0, length_scale_bounds = (1e-1, matern_ub)),
                    1.0 * DotProduct(sigma_0=100.0, sigma_0_bounds = (1e-2, dotProduct_ub)),
                    1.0 * WhiteKernel()])

    def gp_load_data(self, date_obj=None):
        if not date_obj:
            self.trainY = self.residual
        else:
            self.trainY = self.db.load_swe(date_obj, self.site_name, 'swe_lidar', sensor=True)

    def gp_train(self):
        self.GP = GaussianProcessRegressor(kernel=self.cov, alpha=0.0)
        if len(self.trainX.shape) == 1:
            self.GP.fit(self.trainX[:, np.newaxis], self.trainY)
        else:
            self.GP.fit(self.trainX, self.trainY)

    def gp_predict(self):
        self.testY = np.zeros(len(self.testX_all))
        self.testY[:] = 0.
        temp_res = self.GP.predict(self.testX)
        if self.residual is None:
            temp_res[temp_res <= 0.] = 0.
        self.testY[self.testX_finite_bool] = temp_res

    def update_kNN(self, kNN_regressor):
        idx_list = range(kNN_regressor.num_days)
        for idx in idx_list:
            self.update_residual(kNN_regressor.est_residual[idx])
            self.gp_load_data()
            self.gp_train()
            self.gp_predict()
            self._update_kNN(kNN_regressor, idx)
        kNN_regressor.products.append('kNN_GP')
        kNN_regressor.kNN_update_est_stats()
        kNN_regressor.kNN_update_mean_std()

    def _update_kNN(self, kNN_regressor, idx):
        # Update est_raw_dict and est_dict
        kNN_regressor.est_raw_dict['kNN_GP'][idx] = kNN_regressor.est_raw_dict['kNN'][idx] + \
                                                    self.testY.reshape(kNN_regressor.est_raw_dict['kNN'][idx].shape)
        kNN_regressor.est_dict['kNN_GP'][idx] = kNN_regressor.est_raw_dict['kNN_GP'][idx][kNN_regressor.est_raw_dict['lidar'][idx]>=0.]