
from __future__ import division
import sys, math, operator
import numpy as np
import matplotlib.pyplot as plt

class naiveBayes:
	def __init__(self):
		self.counts = {}
		self.features = {}
		self.likelihoods = {}
		self.test_posteriori = {}
		for i in range(10):
			# dictionary storing counts for each class
			self.counts[i] = 0
			# dictionary storing feature counts for each class
			self.features[i] = {}
			# dictionary storing the likelihoods for each feature in each class
			self.likelihoods[i] = {}
			# dictionary storing the posterior probabilities for each class
			self.test_posteriori[i] = []
		# training set
		self.trainset = []
		# training label
		self.trainlabel = []
		# testing set
		self.testset = []
		# testing label
		self.testlabel = []
		# predicted labels for the testing set
		self.predictLabel = []
		# laplacian parameter
		self.k = 1
		# number of possible values the feature can take on
		self.V = 2

	"""Format the input training data and labels"""
	def getTrainSet(self, training_file, training_label):
		# load training image file
		raw_images = open(training_file, 'r').readlines()
		# load training 
		raw_lables = open(training_label, 'r').readlines()
		#print len(lables), len(images)

		for i in range(len(raw_lables)):
			image_temp = raw_images[i*28:i*28+28]
			for j in range(28):
				image_temp[j] = list(image_temp[j].rstrip('\n'))
				for k in range(len(image_temp[j])):
					if image_temp[j][k] == ' ':
						image_temp[j][k] = 0
					else:
						image_temp[j][k] = 1
			self.trainset.append(image_temp)
			self.trainlabel.append(int(raw_lables[i]))

		#print len(self.trainset)
		#print len(self.trainlabel)

	"""Train the Naive Bayes mdoel"""
	def train(self):
		for k in range(len(self.trainlabel)):
			# get the label
			label = self.trainlabel[k]
			# update the count for this class
			self.counts[label] += 1
			# update the feature counts
			image = self.trainset[k]
			for i in range(len(image)):
				for j in range(len(image[0])):
					feature = (i, j, image[i][j])
					self.features[label][feature] = 1 + self.features[label].get(feature, 0)
		# update the likelihood
		for label in self.likelihoods:
			for feature in self.features[label]:
				self.likelihoods[label][feature] = (self.features[label][feature] + self.k) \
													/ (self.counts[label] + self.k * self.V)
		#print self.likelihoods[8]

	"""Format the input testing data and labels"""
	def getTestSet(self, test_file, test_label):
		# load training image file
		raw_images = open(test_file, 'r').readlines()
		# load training 
		raw_lables = open(test_label, 'r').readlines()

		for i in range(len(raw_lables)):
			image_temp = raw_images[i*28:i*28+28]
			for j in range(28):
				image_temp[j] = list(image_temp[j].rstrip('\n'))
				for k in range(len(image_temp[j])):
					if image_temp[j][k] == ' ':
						image_temp[j][k] = 0
					else:
						image_temp[j][k] = 1
			self.testset.append(image_temp)
			self.testlabel.append(int(raw_lables[i]))

		#print len(self.testset)
		#print len(self.testlabel)

	"""classify a give test image"""
	def classify(self, image):
		posteriori = {}
		train_size = len(self.trainlabel)
		for label in range(10):
			# calculate the prior 
			prior = self.counts[label] / train_size
			posteriori[label] = math.log(prior)
			#posteriori[label] = 0
			# calculate the posteriori for each class
			for i in range(len(image)):
				for j in range(len(image[0])):
					feature = (i, j, image[i][j])
					if feature not in self.likelihoods[label]:
						self.likelihoods[label][feature] = (self.features[label].get(feature,0) + self.k) \
                                                    / (self.counts[label] + self.k * self.V)
					posteriori[label] += math.log(self.likelihoods[label][feature])
			self.test_posteriori[label].append(posteriori[label])
		# get the MAP (maximum a posteriori)
		return max(posteriori.iteritems(), key=operator.itemgetter(1))[0]

	"""get classification rate"""
	def classification_rate(self):
		self.class_rate = {}
		self.test_count = {}
		self.test_count_correct = {}
		for label in range(10):
			self.test_count[label] = 0
			self.test_count_correct[label] = 0
		overall = 0
		for i in range(len(self.testlabel)):
			self.test_count[self.testlabel[i]] += 1
			if self.predictLabel[i] == self.testlabel[i]:
				self.test_count_correct[self.testlabel[i]] += 1
				overall += 1
		for label in range(10):
			self.class_rate[label] = self.test_count_correct[label] / self.test_count[label]
		self.ovaerall_accuracy = overall / len(self.testlabel)

	"""get highest posterori probablility from the test images"""
	def highestPosterori(self):
		for label in range(10):
			print "Most prototypical instance of digit "+str(label)+" is: "
			index = self.test_posteriori[label].index(max(self.test_posteriori[label]))
			print "Test sample from index of "+str(index*28)
			#print "Least prototypical instance of digit "+str(label)+" is: "
			#index1 = self.test_posteriori[label].index(min(self.test_posteriori[label]))
			#print "Test sample from index of "+str(index1*28),", we think it is "+str(self.predictLabel[index1])

	"""get confusion matrix"""
	def confusion_matrix(self):
		matrix = np.zeros((10,10))
		for i in range(len(self.testlabel)):
			matrix[self.testlabel[i], self.predictLabel[i]] += 1
		for label in range(10):
			matrix[label,:] /= self.test_count[label]
		return matrix

	"""Do testing on the testing file"""
	def test(self):
		for image in self.testset:
			self.predictLabel.append(self.classify(image))
		self.classification_rate()
		matrix = self.confusion_matrix()
		print "Per class classification rate:"
		print self.class_rate
		print "Overall accuracy is: %g" %(self.ovaerall_accuracy) 
		print "Confusion matrix is:"
		np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
		print(matrix[:,:]*100)

	"""calculate the odd ratio and plot the image"""
	def odd_ratio(self, class1, class2):
		class1_image = np.zeros((28,28))
		class2_image = np.zeros((28,28))
		odd_ratio_image = np.zeros((28,28))
		for x in range(28):
			for y in range(28):
				feature = (x, y, 1)
				if feature not in self.likelihoods[class1]:
					self.likelihoods[class1][feature] = (self.features[class1].get(feature,0) + self.k) \
                                                    / (self.counts[class1] + self.k * self.V)
				if feature not in self.likelihoods[class2]:
					self.likelihoods[class2][feature] = (self.features[class2].get(feature,0) + self.k) \
                                                    / (self.counts[class2] + self.k * self.V)
				class1_image[x][y] = math.log(self.likelihoods[class1][feature])
				class2_image[x][y] = math.log(self.likelihoods[class2][feature])
				odd_ratio_image[x][y] = class1_image[x][y] - class2_image[x][y]
		"""do the plot"""
		plt.imshow(class1_image)
		plt.colorbar()
		plt.show()
		plt.imshow(class2_image)
		plt.colorbar()
		plt.show()
		plt.imshow(odd_ratio_image)
		plt.colorbar()
		plt.show()

if __name__ == "__main__":
	if len(sys.argv) != 5:
		print "Usage: python part1_1.py training_file training_label test_file test_label"
		sys.exit()

	nbClass = naiveBayes()
	training_file = sys.argv[1]
	training_label = sys.argv[2]
	test_file = sys.argv[3]
	test_label = sys.argv[4]
	nbClass.getTrainSet(training_file, training_label)
	nbClass.train()
	nbClass.getTestSet(test_file, test_label)
	nbClass.test()
	nbClass.highestPosterori()
	nbClass.odd_ratio(8,9)
