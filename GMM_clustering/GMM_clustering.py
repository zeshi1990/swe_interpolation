from gdal import gdalconst
import osgeo.osr as osr
from sklearn import mixture
import itertools
import pickle
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import axes_make_locatable
import matplotlib as mpl
mpl.rc("font", family="Helvetica")
mpl.rc("font", size=12)

class GMM_clustering:
    
    def __init__(self, site_name, db):
        # Define parameters needed for GMM clustering analysis
        self.site_name = site_name
        self.db = db
        self.cv_types = ['tied', 'diag', 'full']
        self.color_iter = itertools.cycle(['k', 'r', 'g', 'b', 'c', 'm', 'y'])
        self.n_components_range = range(1, 51)
        self.clf = None
        self.bic = []
        self.class_image = None
        self.sensor_idx = None
    
    # Calculate bic of GMM at different n_components
    def GMM_number(self):
        lowest_bic = np.infty
        GMM_feature = self.db.load_features(sensor_feature=False, exclude_null=True, basin=self.site_name)
        X = np.copy(GMM_feature)
        for cv in self.cv_types:
            print cv
            for n_components in self.n_components_range:
                gmm = mixture.GMM(n_components=n_components, covariance_type=cv, n_init=20, n_iter=15000, random_state=20)
                gmm.fit(X)
                self.bic.append(gmm.bic(X))
                if self.bic[-1] < lowest_bic:
                    lowest_bic = self.bic[-1]
                    self.clf = gmm
        self.bic = np.array(self.bic)
    
    # Plot the bic scores with different GMM n_components
    def GMM_bic_scores(self):
        # predefine some 
        n_components_range, cv_types, color_iter = GMM_priors()
        bars = []

        # generate figure
        plt.figure(figsize=(10, 5))
        spl = plt.subplot(1, 1, 1)
        for i, (cv, color) in enumerate(zip(self.cv_types, self.color_iter)):
            xpos = np.array(self.n_components_range) + .2 * (i - 2)
            bars.append(plt.bar(xpos, self.bic[i * len(self.n_components_range):
                                          (i + 1) * len(self.n_components_range)],
                                width=.2, color=color))
        plt.xticks(self.n_components_range)
        plt.ylim([self.bic.min() * 1.01 - .01 * self.bic.max(), self.bic.max()])
        plt.title('BIC score per model')
        xpos = np.mod(self.bic.argmin(), len(self.n_components_range)) + .65 +\
            .2 * np.floor(self.bic.argmin() / len(self.n_components_range))
        plt.text(xpos, self.bic.min() * 0.97 + .03 * self.bic.max(), '*', fontsize=14)
        spl.set_xlabel('Number of components')
        spl.legend([b[0] for b in bars], self.cv_types)
        plt.savefig("GMM_bic_scores_" + self.site_name.lower() + ".pdf")
        plt.show()
        
    # From the figure shown above, we could see that the number of 
    # components with minimum BIC score is 27 with full covariance matrix. 
    # So in the next step we need to include ```components = 27``` in the GMM parameters
    def GMM_loc(self, elev_cut = None):
        """
        input: elev_cut, int
        output: class_image, 2D array the same shape as DEM, showing the classes of each pixel belongs to
                sensor_idx, sensor's [row, col] indices on the map
        """
        GMM_feature = self.db.load_features(sensor_feature=False, exclude_null=False, basin=self.site_name)
        dem = self.db.query_map("DEM", "topo", self.site_name)
        dem[dem < 0.] = np.nan
        if elev_cut is not None:
            print "Enforce more sampling"
            GMM_feature_idx = np.where(np.logical_and(np.isfinite(GMM_feature[:, 2]), 
                                                                  GMM_feature[:, 2] >= elev_cut))
        else:
            GMM_feature_idx = np.where(np.isfinite(GMM_feature[:, 2]))

        # Extract finite feature space
        GMM_feature_finite = GMM_feature[GMM_feature_idx]
        X = GMM_feature_finite

        # Initialize class image for entire map space, including nans
        class_image = np.zeros(len(GMM_feature))
        class_image[np.isnan(GMM_feature[:, 2])] = np.nan

        # fit and predict GMM classification
        labels = clf.predict(X)

        # Labeling the class to the finite/valid features
        class_image[GMM_feature_idx] = labels
        class_image = np.reshape(class_image, dem.shape)

        # Loop through labels to get the centroid of each cluster and find the nearest neighbor in 2D space
        unique_labels = np.unique(labels)
        x_idx_list = []
        y_idx_list = []
        for label in unique_labels:
        #     print label
            temp_labels_idx = np.where(labels==label)
            X_label = X[labels==label, 2:]
            X_label_KDTree = KDTree(X_label)
            X_label_median = np.nanmean(X_label, axis=0)
            dist, idx = X_label_KDTree.query(np.array([X_label_median]))
            X_label_median_idx = temp_labels_idx[0][idx]
            GMM_feature_median_label_idx = GMM_feature_idx[0][X_label_median_idx]
            x_idx = xgrid.flatten()[GMM_feature_median_label_idx]
            y_idx = ygrid.flatten()[GMM_feature_median_label_idx]
            x_idx_list.append(x_idx[0][0])
            y_idx_list.append(y_idx[0][0])
        self.sensor_idx = (np.array(y_idx_list), np.array(x_idx_list))
        self.class_image = class_image
        
    # Plot final GMM location maps over DEM of the basin
    def GMM_loc_map(self):
        # from mpl_toolkits.axes_grid1 import make_axes_locatable
        fig, ax = plt.subplots(figsize=(3.5, 3.5))

        # Load basin
        dem = self.db.query_map('DEM', 'topo', self.site_name)
        dem_show = np.copy(dem)
        dem_show[dem_show <= 0.] = np.nan
        im = ax.imshow(dem_show, cmap='terrain', interpolation='bilinear')
        x_idx_list = self.sensor_idx[1]
        y_idx_list = self.sensor_idx[0]
        x_idx_list_show = np.array(x_idx_list) + 0.5
        y_idx_list_show = np.array(y_idx_list) + 0.5
        plt.plot(x_idx_list_show, y_idx_list_show, '.k', markersize=10)
        plt.xlim([0, dem.shape[1]])
        plt.ylim([dem.shape[0], 0])
        plt.axis('off')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="5%", pad=0.05)
        cb = fig.colorbar(im, cax=cax, ticks=[1200, 2000, 2800, 3600], orientation='horizontal')
        cb.outline.set_visible(False)
        plt.subplots_adjust(hspace=0)
        plt.savefig("selected_location_" + self.site_name.lower() + ".pdf", dpi=1000, bbox_inch="tight")
