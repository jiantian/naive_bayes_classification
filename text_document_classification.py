
from __future__ import division
import sys, math, operator, collections
import numpy as np

class naiveBayes:
	def __init__(self):
		self.counts = {}
		self.features = {}
		self.likelihoods = {}
		self.msg_counts = {}
		self.classes = [0, 1, 2, 3, 4, 5, 6, 7]
		for i in self.classes:
			# dictionary storing counts for each class
			self.counts[i] = 0
			# dictionary storing feature counts for each class
			self.features[i] = {}
			# dictionary storing the likelihoods for each feature in each class
			self.likelihoods[i] = {}
			# dictionary storing the message counts for each class
			self.msg_counts[i] = 0
		# testing label
		self.testlabel = []
		# predicted labels for the testing set
		self.predictLabel = []
		# laplacian parameter
		self.k = 1

	def train(self, training_file):
		lines = open(training_file, "r").readlines()
		# first, get the dictionary as well as the word count in each class
		for line in lines:
			line = line.strip().split(' ')
			label = int(line.pop(0))
			self.msg_counts[label] += 1
			for item in line:
				item = item.split(':')
				word = item[0]
				count = int(item[1])
				self.counts[label] += count
				self.features[label][word] = self.features[label].get(word,0) + count
				for other_label in self.classes:
					if other_label != label:
						if word not in self.features[other_label]:
							self.features[other_label][word] = 0
		# next, calculate the likelihood
		for label in self.likelihoods:
			V = len(self.features[label])
			for word in self.features[label]:
				self.likelihoods[label][word] = (self.features[label][word] + self.k) \
												/ (self.counts[label] + self.k * V)

	"""do classification of a single line in the testing file"""
	def classify(self, line):
		line = line.strip().split(' ')
		label = int(line.pop(0))
		self.testlabel.append(label)
		posteriori = {}
		total_msg = sum(self.msg_counts.values())
		for label in self.classes:
			# calculate the prior
			prior = self.msg_counts[label] / total_msg
			# calculate the posterior in each class
			posteriori[label] = math.log(prior)
			for item in line:
				item = item.split(':')
				word = item[0]
				count = int(item[1])
				# if word in the test documents does not occur in the dictionary, ignore it 
				if word in self.likelihoods[label]:
					posteriori[label] += count * math.log(self.likelihoods[label][word])
		return max(posteriori.iteritems(), key=operator.itemgetter(1))[0]

	"""do testing, print the results"""
	def test(self, test_file):
		lines = open(test_file, 'r').readlines()
		for line in lines:
			self.predictLabel.append(self.classify(line))
		self.classification_rate()
		print "Per class classification rate:"
		print self.class_rate
		print "Overall accuracy is: %g" %(self.overall_accuracy)
		matrix = self.confusion_matrix()
		print "Confusion matrix is:"
		np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
		print(matrix[:,:]*100)
		# return top 20 words with the highest likelihood in each class
		for label in self.classes:
			d = collections.Counter(self.likelihoods[label])
			print "Top 20 words words with the highest likelihood in class "+str(label)+":"
			for k, v in d.most_common(20):
				print '%s' % (k)

	"""get the per-class and overall accuracy"""
	def classification_rate(self):
		self.class_rate = {}
		self.test_count = {}
		self.test_count_correct = {}
		for label in self.classes:
			self.test_count[label] = 0
			self.test_count_correct[label] = 0
		overall = 0
		for i in range(len(self.testlabel)):
			self.test_count[self.testlabel[i]] += 1
			if self.predictLabel[i] == self.testlabel[i]:
				self.test_count_correct[self.testlabel[i]] += 1
				overall += 1
		for label in self.classes:
			self.class_rate[label] = self.test_count_correct[label] / self.test_count[label]
		self.overall_accuracy = overall / len(self.testlabel)

	"""get confusion matrix"""
	def confusion_matrix(self):
		dim = len(self.classes)
		matrix = np.zeros((dim,dim))
		for i in range(len(self.testlabel)):
			matrix[self.testlabel[i], self.predictLabel[i]] += 1
		for label in self.classes:
			matrix[label,:] /= self.test_count[label]
		return matrix

	"""get the log-odd ratio and display top 20 words with highest log-odd ratio"""
	def log_odd_ratio(self, class1, class2):
		odd_ratio = {}
		for word in self.likelihoods[class1]:
			odd_ratio[word] = math.log(self.likelihoods[class1][word] / self.likelihoods[class2][word])
		d = collections.Counter(odd_ratio)
		print "Top 20 words with highest log-odd ratio:"
		for k, v in d.most_common(20):
			print '%s' % (k)

	"""output text for word cloud"""
	def wordCloudText(self):
		for label in self.classes:
			out_file_name = "word_cloud_class"+str(label)+".txt"
			f = open(out_file_name, 'w')
			for word, weight in self.likelihoods[label].iteritems():
				f.write(word+":"+str(weight)+"\n")
			f.close()

if __name__ == "__main__":
	if len(sys.argv) != 3:
		print "Usage: python part2.py training_file test_file"
		sys.exit()

	nbClass = naiveBayes()
	training_file = sys.argv[1]
	test_file = sys.argv[2]

	nbClass.train(training_file)
	nbClass.test(test_file)
	nbClass.log_odd_ratio(5,7)
	nbClass.wordCloudText()
