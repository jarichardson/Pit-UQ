#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 20:55:50 2021

@author: Jacob Richardson






Here are some variables to change. First, the input and output filenames.
Then, a safe guess between the x-resolution interval and the full x-range
Finally, unit conversion labels and factors.
"""

#NEPTEC DATA
importFilename = '../data/MicroclinePit21S1mmx1mmRaw.csv'
#CSV OUTPUT FILE
listExportFilename = '../data/Microcline_P21_neptec_1mm.csv'

#SAFE GUESS >xResolution; << xRange
min_row_step = 0.2

#DESIRED OUTPUT UNIT LABEL
lunits = 'um'
#MULTIPLICATION TO CONVERT TO DESIRED UNITS
unit_conversion_factor = 1000.0 #1000 assumes raw data in mm, desired data in um



'''
The code is below
'''

import numpy as np
import matplotlib.pyplot as plt


### Read the data

csv_data = np.loadtxt(importFilename,delimiter=',')

#print the first 10 records
print('The first 10 records look like:')
print(csv_data[0:10])

x_data = csv_data[:,0]
y_data = csv_data[:,1]
z_data = csv_data[:,2]

print('')
print('In total, there are %d coordinates' % len(csv_data))
print('The x and y data ranges are: (%0.6f, %0.6f), (%0.6f, %0.6f)' % (np.min(x_data),np.max(x_data),np.min(y_data),np.max(y_data)))
print('The z data range is: (%0.6f, %0.6f)' % (np.min(z_data),np.max(z_data)))
print('90%% of the heights are between %0.6f and %0.6f' % (np.percentile(z_data,5),np.percentile(z_data,95)))

lx_data = csv_data[:,0]
ly_data = csv_data[:,1]
lz_data = csv_data[:,2]

plt.clf()
msize = csv_data[:,2]*-1 + 8
plt.scatter(lx_data,ly_data,c=lz_data)
plt.clim([np.percentile(lz_data,10),np.percentile(lz_data,90)])
plt.colorbar()
plt.show()

### Estimate number of columns with a histogram

#calculate the difference between x[i] and x[i-1]. 
#The number should usually be small... except at new lines!
x_diff = np.zeros(len(lx_data)-1)
for i,x in enumerate(lx_data):
    if i>0:
        x_diff[i-1] = abs(lx_data[i] - lx_data[i-1])

plt.clf()
plt.hist(x_diff)
plt.show()


#How many dx are above 0.2? This is how many 'new rows' are in the dataset
#The first row isn't counted here, so we add 1 to the solution
rowCt = len(np.where(x_diff > min_row_step)[0]) + 1

print('This dataset has %d coordinates' % len(lx_data))

print('This dataset has %0.6f rows.' % rowCt)

colCt = len(lx_data) / rowCt
print('This dataset has %0.6f columns.' % colCt)

xRes = (np.max(lx_data) - np.min(lx_data))/colCt
yRes = (np.max(ly_data) - np.min(ly_data))/rowCt

print('The average x interval is: %0.6f' % xRes)
print('The average y interval is: %0.6f' % yRes)


### Visualize grid to verify regridding!

#reshape the data from a list to a row x column array
#the [::-1] reverses the array, so the first line is the highest line
lZZ = lz_data.reshape(int(rowCt),int(colCt))[::-1]

#illustrate it as a raster with imshow
lmap_extent = [np.min(lx_data),np.max(lx_data),np.min(ly_data),np.max(ly_data)]
plt.clf()
plt.imshow(lZZ,extent=lmap_extent)
plt.clim([np.percentile(lz_data,10),np.percentile(lz_data,90)])
plt.colorbar()
plt.show()

### Convert data to desired Unit
lx_data *= unit_conversion_factor
ly_data *= unit_conversion_factor
lz_data *= unit_conversion_factor
xRes *= unit_conversion_factor
yRes *= unit_conversion_factor



### Export data 


#From visual inspection the zRes from this data looks to be 0.1 um.
#Perhaps autocheck for this in the future
zRes = 0.1

with open(listExportFilename, 'w') as ef:
    ef.write('# Produced from Neptec List: %s\n' % importFilename)
    ef.write('# Units: %s\n' % lunits)
    ef.write('# x interval: %0.6f\n' % xRes)
    ef.write('# y interval: %0.6f\n' % yRes)
    ef.write('# rows: %d\n' % rowCt)
    ef.write('# columns: %d\n' % colCt)
    ef.write('# vertical resolution: %0.6f\n' % zRes)
    ef.write('# \n# x, y, z\n')
    
    #Now export the data!
    for i in range(len(lz_data)):
        ef.write('%0.8f,%0.8f,%0.8f\n' % (lx_data[i],ly_data[i],lz_data[i]))
        
#Here's an example of what the data look like:
print('Example output:')
for i in range(10):
    print('%0.8f,%0.8f,%0.8f' % (lx_data[i],ly_data[i],lz_data[i]))