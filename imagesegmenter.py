import sys
from PIL import Image
from sklearn import datasets
import numpy


# Implement your own version of K-Means and use it to cluster the data (i.e. the features for each pixel) into K clusters. If a cluster is ever empty during the procedure, assign a random data point to it. Use random initializations for the cluster centers, and iterate until the centroids converge.

def my_kmeans(k, pixels, lambda_iter):
    rand_centers = pixels[numpy.random.choice(pixels.shape[0], k, replace=False)]
    for i in range(lambda_iter):
        centers = rand_centers
        energies = []

        ps0 = pixels.shape[0]
        for x in range(ps0):
            cur_energies = []
            for j in range(k):
                cur_energies.append(energyCalculation(pixels[x, :], centers[j]))
            energies.append(cur_energies)
        best_energies = []

        ps0 = pixels.shape[0]
        for x in range(ps0):
            best_energies.append(numpy.argmin(energies[x]))
        whichPixel = numpy.hstack((numpy.asarray(best_energies).reshape((numpy.asarray(best_energies).shape[0],1)), pixels))

        mean_energies = []
        for x in range(k):
            mean_energies.append(numpy.mean(whichPixel[whichPixel[:, 0]==x], axis=0))
        means = numpy.zeros((1, whichPixel.shape[1]))

        for x in mean_energies:
            means = numpy.vstack((means, numpy.array(x)))
        rand_centers = numpy.delete(numpy.delete(means, (0), axis=0), 0, axis=1)

        threshold = 1e-4
        if energyCalculation2(centers, rand_centers) < threshold:
            all_pixels = []
            for x in range(k):
                all_pixels.append(whichPixel[whichPixel[:, 0]==x])
            return (all_pixels, mean_energies)
        all_pixels = []
        for xx in range(k):
            all_pixels.append(whichPixel[whichPixel[:, 0]==xx])

        clustered = (all_pixels, mean_energies)
        return clustered

def energyCalculation(point, cluster_center):
    # energy calculation data points - center sum squared
    return sum((point -  cluster_center)**2)

def energyCalculation2(centers1, centers2):
    # energy calculation data points - center sum squared
    return sum(sum((centers1 - centers2)**2))

k = int(sys.argv[1])
in1 = sys.argv[2]
out1 = sys.argv[3]

in1 = Image.open(in1)
in1_load = in1.load()

# converted the input image into a data set with five features 
widthx, heighty = in1.size
numpy_pixels = []
for i in range(widthx):
    for j in range(heighty):
        lay0 = in1_load[i,j][0]
        lay1 = in1_load[i,j][1]
        lay2 = in1_load[i,j][2]
        numpy_pixels.append((lay0, lay1, lay2, i, j))

numpy_pixels = numpy.array(numpy_pixels).astype(float)

# standardize the values of each feature in the data set to improve results
mean_std_features = []
for j in range(numpy_pixels.shape[1]):
    mean = numpy.mean(numpy_pixels[:,j])
    stdev = numpy.std(numpy_pixels[:,j])
    for i in range(numpy_pixels.shape[0]):
        # subtract mean & divide by stdev
        numpy_pixels[i, j] = (numpy_pixels[i, j] - mean) / stdev
    stdfeature=(mean,stdev)
    mean_std_features.append(stdfeature)
mean_std_features = numpy.asarray(mean_std_features).astype(float)
p, centers = my_kmeans(k, numpy_pixels, 120)
centers_list = []
for x in range(k):
    center = numpy.array(centers[x])
    center = center.reshape(1, 6)
    center = center[:, 1:4]
    centers_list.append(center)
data_centers = []
for x in range(k):
    data_centers.append(numpy.array(p[x])[:, 1:6])
for x in range(k):
    data_centers[x][:, 0:3] = centers_list[x]
pixel_data = numpy.zeros((1,5))
for x in range(k):
    pixel_data = numpy.vstack((pixel_data,data_centers[x]))
pixel_data = pixel_data[1:(in1.size[0] * in1.size[1] + 1),:]

# Create an output image the same size as the input image. Then fill in the color values of each pixel of the image based on the informs us that the pixel at (ip,jp) should have color (rC(p),gC(p),bC(p)). Note that you also have to undo the feature standardization at this point (just invert the standardization equation by solving for the original value given the standardized value).
for col in range(pixel_data.shape[1]):
    mean = mean_std_features[col][0]
    std = mean_std_features[col][1]
    for row in range(pixel_data.shape[0]):
        pixel_data[row,col] = numpy.int(numpy.around((pixel_data[row,col] * std) + mean))
needed_layers = numpy.asarray(sorted(pixel_data, key=lambda x:(x[3], x[4])))[:, 0:3].astype(int)
to_print = [tuple(i) for i in needed_layers]
dimensions = (in1.size[0], in1.size[1])
output = Image.new('RGB', dimensions, "white")
grid = output.load()

x = 0
for i in range(output.size[0]):
    for j in range(output.size[1]):
        grid[i,j] = to_print[x]
        x += 1
output.save(out1)