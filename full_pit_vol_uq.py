#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 14:39:23 2021

@author: Jacob Richardson




Here are some input parameters for the user to change as needed. First is the 
path to the data. The * variables below will need iterating to get good results.
So guess, run, correct, rerun.
"""
filepath = 'data/FC3_P17_keyence.csv'

### pre-detrend variables
#pre-detrend pit height guess
pit_margin_first_guess = 325 # *

### post-detrend variables
#rectangular mask for UQ estimation
rect_mask_xmin = 200 # *
rect_mask_xmax = 850 # *
rect_mask_ymin = 125 # *
rect_mask_ymax = 675 # *

#how many standard deviations off of the surface height do you define as "the pit"
pit_sigma_threshold = 2

#Final plot. Make big enough for contour to be seen. Just a visual thing
contour_color_add = 50
contour_width_add = 5 #5 for keyence, 1 for neptec probably


'''
Code is below. First are some functions, then the main code below that
'''

import numpy as np
import matplotlib.pyplot as plt
import sys, cv2
import scipy.linalg as la


class metadata():
	def __init__(self):
		self.filename = '' # path to provenance of metadata
		self.xRes	= 0 # x-resolution/interval
		self.yRes = 0 # y-resolution/interval
		self.zRes	= 0 # z-resolution/interval
		self.rows	= 0 # number of rows in the data
		self.cols  = 0 # number of columns in the data
		self.units  = '' # string of units in file, e.g. um, mm, nm
		self.totLines = 0 # Total number of lines in the file
		
	def read_unified_csv_metadata(self):
		'''
		Parameters
		----------
		filepath : STRING
			Filepath to a CSV with minimal metadata. This CSV format can be created
			with Keyence or Neptec data using the Loading Height Data notebook.
	
		Returns
		-------
		Metadata of the CSV file
		'''
		with open(self.filename) as kf:
			line = kf.readline()
			self.totLines = 0
			while line:
				linestr = line.split(':')
				metadata_field = linestr[0]
				if 'x interval' in metadata_field:
					self.xRes	  = float(linestr[1])
				elif 'y interval' in metadata_field:
					self.yRes	  = float(linestr[1])
				elif 'vertical resolution' in metadata_field:
					self.zRes	  = float(linestr[1])
				elif 'rows' in metadata_field:
					self.rows	  = int(linestr[1])
				elif 'columns' in metadata_field:
					self.cols	  = int(linestr[1])
				elif 'Units' in metadata_field:
					self.units	 = linestr[1].strip()
				self.totLines += 1
				line = kf.readline()#.strip()
	
		
	def get_metadata(self, filepath):
		self.filename = filepath
		#for now the only functionality is to read the "unified" style of csv
		self.read_unified_csv_metadata()
		#A simple metadata check can be implemented if need be. Just check to
		#Make sure the variables aren't still 0.
		#ret = self.check_metadata()
		
		#Get headerlines to return
		h_lines = self.totLines - self.rows*self.cols
		return h_lines
	
'''
THE MAIN CODE
'''

print('Loading Metadata\n')
### Load the metadata
md = metadata()
header_lines = md.get_metadata(filepath)
#minimal check for success. If fails, exit
if header_lines < 1:
	sys.exit(0)
	
### Load the real data
print('Loading Array Data\n')
csv_data = np.loadtxt(md.filename,skiprows=header_lines,delimiter=',')

x_data = csv_data[:,0]
y_data = csv_data[:,1]
z_data = csv_data[:,2]

#reshape the data into row x column arrays
XX = x_data.reshape(md.rows,md.cols)
YY = y_data.reshape(md.rows,md.cols)
ZZ = z_data.reshape(md.rows,md.cols)
ZZ = ZZ[::-1] #reverses it so the top row corresponds to the highest Y value
#This is basically a vertical reflection to correct for the way reshaping is done.

#make an extent array for all imshow plotting calls later
map_extent = [np.min(XX),np.max(XX),np.min(YY),np.max(YY)]

#plot the raw data
plt.clf()
plt.imshow(ZZ,extent=map_extent)
plt.xlabel('%s' % md.units)
plt.ylabel('%s' % md.units)
plt.colorbar(label=('%s' % md.units))
plt.title('Measured Height Data')
plt.show()

#make a very large number of bins for the histogram to estimate pit depth
bins = np.linspace(np.min(z_data),np.max(z_data),100)

#Plot the CDF
plt.clf()
plt.hist(z_data, bins=bins)
plt.xlabel('%s' % md.units)
plt.ylabel('pixel count')
plt.title('histogram of height values')
plt.show()

### Detrend the data
print('Detrending the Surface...\n')

#Get just the non-pit values
'''pit_margin_first_guess is estimated at the top of the file '''
surf_xs = x_data[np.where(z_data>pit_margin_first_guess)]
surf_ys = y_data[np.where(z_data>pit_margin_first_guess)]
surf_zs = z_data[np.where(z_data>pit_margin_first_guess)]

#Plot this data subset
plt.clf()
plt.scatter(surf_xs,surf_ys,c=surf_zs)
plt.title('pitless surface for plane fitting')
plt.colorbar()
plt.show()

#First Concatenate the left side of the equation (Ax)
pitA = np.c_[surf_xs,surf_ys,np.ones(len(surf_xs))]

#perform least squares inversion, keep the coefficients, ignore 3 other outputs
Coeffs,_,_,_ = la.lstsq(pitA,surf_zs)

#Now calculate the plane on ALL sample locations, not just the mask!
#In this case, we'll use the gridded XX,YY,ZZ data
PP = Coeffs[0]*XX + Coeffs[1]*YY + Coeffs[2]
ZZ_resid = ZZ - PP

# plot with imshow
plt.clf()
plt.imshow(PP,extent=map_extent)
plt.xlabel('%s' % md.units)
plt.ylabel('%s' % md.units)
plt.colorbar(label=('%s' % md.units))
plt.title('best fit plane')
plt.show()

plt.clf()
plt.imshow(ZZ_resid,extent=map_extent)
plt.xlabel('%s' % md.units)
plt.ylabel('%s' % md.units)
plt.colorbar(label=('%s' % md.units))
plt.title('detrended residual heights')
plt.show()


### Quantify UQ 
print('Quantifying surface uncertainty...\n')

#First construct a boolean grid the same size as the ZZ array
#numpy ones will make all values "True"
mask = np.ones(np.shape(ZZ_resid), dtype=bool)

#Define an interior slice of the array as False
#The geometry of this is user-defined at the top of the code
mask[rect_mask_ymin:rect_mask_ymax,rect_mask_xmin:rect_mask_xmax] = False

ZZ_mask = ZZ_resid * mask

plt.clf()
plt.imshow(ZZ_mask,extent=map_extent)
plt.xlabel('%s' % md.units)
plt.ylabel('%s' % md.units)
plt.colorbar(label=('%s' % md.units))
plt.title('pitless surface (rectangle mask) for UQ')
plt.show()

#This makes a list of Z values not masked out
ZZ_surfonly = ZZ_resid[np.where(mask)]

Z_res_mean = np.mean(ZZ_surfonly)
Z_res_std =  np.std(ZZ_surfonly)

print('The mean height is %0.6f um' % Z_res_mean)
print('The standard deviation of height is %0.6f um' % Z_res_std)


### Define the pit
print('Defining the Pit and measuring!...\n')

#Define the contour below which values represent pits
#pit_sigma_threshold is user-defined at the top of the code
pitContour = Z_res_mean - pit_sigma_threshold  *Z_res_std
print('Anomalously low locations are anywhere below %0.6f um' % pitContour)

#make a boolean pit map. Set all values to false (not a pit) with np zeros
pitmap = np.zeros(np.shape(ZZ_resid), dtype='uint8')

#Anywhere where the residual Z value is less than the pit contour is a pit
pitmap[np.where(ZZ_resid<pitContour)] = True

plt.clf()
plt.imshow(pitmap, interpolation='none',extent=map_extent)
plt.xlabel('%s' % md.units)
plt.ylabel('%s' % md.units)
plt.colorbar(label=('anomalously low = 1'))
plt.title('Pits are yellow')
plt.show()


# Raster analysis
# Read the bitmap
# We use the binary pitmap above
# convert it to a full contrast "grey scale" image. This makes it easier for the raster module openCV
im = pitmap*255

# Now create some contours

# Generate intermediate image; use morphological closing to keep contiguous parts together
inter = cv2.morphologyEx(im, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

# Find largest contour in intermediate image
cnts, _ = cv2.findContours(inter, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cnt = max(cnts, key=cv2.contourArea)

# "Paint in the lines": Make a map of 0s and change them to 1s if they are within the contour
final_mask = np.zeros(im.shape, np.uint8)
cv2.drawContours(final_mask, [cnt], -1, 1, cv2.FILLED)
final_mask = cv2.bitwise_and(im, final_mask)

print('Apply Volumizing Mask to Accentuating Pore')
# Plot the pit mask
plt.clf()
plt.imshow(final_mask, interpolation='none',extent=map_extent)
plt.xlabel('%s' % md.units)
plt.ylabel('%s' % md.units)
plt.colorbar(label=('the pit is yellow'))
plt.title('The largest pit identified')
plt.show()


#if each pit pixel is a 1, then the sum of the mask will be the pit size in pixels
pit_area = np.sum(final_mask)
print("The pit is %0.1f%% of the total map area" % (pit_area/np.size(final_mask)*100))

### Final Volume and Uncertainty Calculations
# Capture the raw depths of all values within the pit
# Also subtract from these raw values the average surface height calculated previously without the pit. 
#  Alternatively, this subtraction can be performed earlier as a second step to the detrending process.
# Then multiply by -1 to make the depths positive values
pit_depths = (ZZ_resid - Z_res_mean) * final_mask * -1

### Now sum the depths to create an integrated depth
total_depth = np.sum(pit_depths)
max_depth = np.max(pit_depths)
med_depth = np.median(pit_depths[np.where(pit_depths>0)])

#Multiply by spatial resolution for the volume
total_volume = abs(total_depth * md.xRes * md.yRes)

print('The Total Volume of the pit is %0.3f cu. %s' % (total_volume, md.units))

#Modified From Section 2 above:
#Calculate total area in the right units
total_area = md.xRes * md.yRes * pit_area

#Make an array, the same size as the cells in question, and make all array values = the standard deviation
uncertainty_array = np.ones(pit_area) * Z_res_std

#Add these uncertantites in quadrature 
depth_uncertainty = np.sum(uncertainty_array**2)**0.5

#Multiply the height uncertainty by the grid cell area to get volume
total_uncertainty = md.xRes * md.yRes * depth_uncertainty

print('Vertical uncertainty has been previously estimated to be +/- %0.3f %s/px' % (Z_res_std, md.units))
print('The total pit area is %0.3f sq. %s' % (total_area, md.units))
print('Over this area, the Total Volume Uncertainty is %0.3f cu. %s' % (total_uncertainty, md.units))
print('This uncertainty is %0.3f%% of the total volume' % (total_uncertainty/total_volume*100) )






print('\n\n\n   Final Readout\n\n')
print('Calculated from file: %s' % md.filename)
print('Pit Area:             %0.6f sq. %s' % (total_area, md.units))
print('Max Pit Depth:        %0.6f %s' % (max_depth, md.units))
print('Median Pit Depth:     %0.6f %s' % (med_depth, md.units))
print('Pit Volume:           %0.6f +/- %0.6f cu. %s' % (total_volume, total_uncertainty, md.units))
print('Estimated Vertical Uncertainty: +/- %0.6f %s/px' % (Z_res_std, md.units))
print('Uncertainty as a percent of Total Volume: %0.6f%%' % (total_uncertainty/total_volume*100) )
print('')
print('   Extra script parameters')
print('sigma threshold: %0.2f' % pit_sigma_threshold)
print('pre-detrend pit height estimate: %0.6f' % pit_margin_first_guess)
print('UQ rectangular pit mask array: [%d:%d,%d:%d]' % 
	  (rect_mask_xmin,rect_mask_xmax,rect_mask_ymin,rect_mask_ymax))

#Make a plot of the raw data with an outline of the pit.
canvas = np.zeros_like(im)
cv2.drawContours(canvas , cnt, -1, contour_color_add, contour_width_add)

plt.clf()
plt.imshow(canvas+ZZ, interpolation='none',extent=map_extent)
plt.clim([np.max(ZZ)-1.5*max_depth,np.max(canvas+ZZ)]) #a nice color ramp
plt.xlabel('%s' % md.units)
plt.ylabel('%s' % md.units)
plt.colorbar(label=('%s' % md.units))
plt.title('Pit Location in Original Data with synthetic pit boundary')
plt.show()