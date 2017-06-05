import numpy as np 
import sys
model = {}

#got the code to download the glove file in this StackOverflow post: 
#https://stackoverflow.com/questions/37793118/load-pretrained-glove-vectors-in-python
def loadGloveModel(gloveFile):
    print "Loading Glove Model"
    f = open(gloveFile,'r')
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        print word
        embedding = [float(val) for val in splitLine[1:]]
        model[word] = embedding
    print "Done. ",len(model)," words loaded!"

def create_glove(train):
	first = 0
	count = 0
	for word in train:
		glove = model.get(word, None)
		if (glove != None):
			glove = np.asarray(glove)
			count += 1
			if (first == 0):
				first = 1
				add = glove
			else:
				add = np.add(add, glove)
	average_glove = np.divide(add, count)
	return average_glove



if __name__ == "__main__":
	glove_matrix = []
	loadGloveModel("glove.6B.100d.txt")
	train_matrix = sys.argv[1]
	for train in train_matrix:
		glove_matrix.append(create_glove(train))
	return glove_matrix



