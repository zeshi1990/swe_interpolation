from gdal import gdalconst
import osgeo.osr as osr
import gdal
import numpy as np
site_name_abbr = {"Merced": "mb", "Tuolumne":"tb"}

class rasterTools:
	def __init__(self):
		pass
	def gen_snodas_fn(self, date):
		folder_name = "SNODAS/"
		fn = folder_name + \
			"us_ssmv11034tS__T0001TTNATS" + \
			str(date.year) + \
			str(date.month).zfill(2) + \
			str(date.day).zfill(2) + \
			"05HP001.Hdr"
		return fn

	def reproject_snodas(self, dates, site_name):
		match_fn = "ASO_Lidar/" + site_name + "_500m_DEM.tif"
		for temp_date in dates:
			src_fn = self._gen_snodas_fn(temp_date)
			dst_fn = "SNODAS/" + site_name_abbr[site_name].upper() + "_" + temp_date.strftime("%Y%m%d") + ".tif"
			self.reproject(fn_src, match_fn, dst_fn)
			
	def reproject(self, src_fn, match_fn, dst_fn, lidar=False):
		src_filename = src_fn
		src = gdal.Open(src_filename, gdalconst.GA_ReadOnly)
		src_proj = src.GetProjection()
		src_geotrans = src.GetGeoTransform()

		# We want a section of source that matches this:
		match_filename = match_fn
		match_ds = gdal.Open(match_filename, gdalconst.GA_ReadOnly)
		match_proj = match_ds.GetProjection()
		match_geotrans = match_ds.GetGeoTransform()
		wide = match_ds.RasterXSize
		high = match_ds.RasterYSize

		# Output / destination
		dst_filename = dst_fn
		dst = gdal.GetDriverByName('GTiff').Create(dst_filename, wide, high, 1, gdalconst.GDT_Float32)
		dst.SetGeoTransform( match_geotrans )
		dst.SetProjection( match_proj)
		if not lidar:
			gdal.ReprojectImage(src, dst, src_proj, match_proj, gdalconst.GRA_Bilinear)
		else:
			gdal.ReprojectImage(src, dst, src_proj, match_proj, gdalconst.GRA_Average)
		del dst
	    
	def reproject_lidar(self, src_fn, match_fn, dst_fn):
		self.reproject(src_fn, match_fn, dst_fn, lidar=True)

	def array_to_raster(array, dst_fn, match_fn):
		"""Array > Raster
		Save a raster from a C order array.

		:param array: ndarray
		:param dst_fn: destination file name
		:param match_fn: the file name that having the matching projection / transformation
		"""
		match_ds = gdal.Open(match_fn, gdalconst.GA_ReadOnly)
		wkt_geotransform = match_ds.GetGeoTransform()
		wkt_projection = match_ds.GetProjection()

		# You need to get those values like you did.
		raster_shape = array.shape
		x_pixels = raster_shape[1]  # number of pixels in x
		y_pixels = raster_shape[0]  # number of pixels in y
		PIXEL_SIZE_X = wkt_geotransform[1]  # size of the pixel...        
		x_min = wkt_geotransform[0]  
		y_max = wkt_geotransform[3]  # x_min & y_max are like the "top left" corner.
		PIXEL_SIZE_Y = wkt_geotransform[5]


		driver = gdal.GetDriverByName('GTiff')

		dataset = driver.Create(
		dst_fn,
		x_pixels,
		y_pixels,
		1,
		gdal.GDT_Float32, )

		dataset.SetGeoTransform((
		x_min,    # 0
		PIXEL_SIZE_X,  # 1
		0,                      # 2
		y_max,    # 3
		0,                      # 4
		PIXEL_SIZE_Y))  

		dataset.SetProjection(wkt_projection)
		dataset.GetRasterBand(1).WriteArray(array)
		dataset.FlushCache()  # Write to disk.
		return dataset, dataset.GetRasterBand(1)

	def reproject_all_recon(site_name='Merced'):
		year_list = range(2001, 2015)
		match_fn = "ASO_Lidar/" + site_name + "_500m_DEM.tif"
		dst_dir = site_name_abbr[site_name].upper() + "_recon/"
		for temp_year in year_list:
			print temp_year
			temp_date = date(temp_year, 3, 1)
			end_date = date(temp_year, 8, 31)
			while temp_date <= end_date:
				src_fn = str(temp_year) + "/" + temp_date.strftime("%d%b%Y").upper() + ".tif"
				dst_fn = dst_dir + temp_date.strftime("%d%b%Y").upper() + ".tif"
				self.reproject(src_fn, match_fn, dst_fn)
				temp_date += timedelta(days=1)
