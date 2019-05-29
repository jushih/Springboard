#!/Users/julieshih/anaconda/bin/python
from sklearn.neighbors.unsupervised import NearestNeighbors

def fit_knn(n=5,encodings): 
    nbrs =  NearestNeighbors(n_neighbors=n).fit(encodings)
    return nbrs

def retrieve_img(knnModel,img):
    distances, indices = knnModel.kneigbhors(img)
    return distances, indices

#class NearestNeighbors:

    # Implements a k-Nearest Neighbor Model.
#    def __init__(self,n_neighbors=5):
#        self.n_neighbors = n_neighbors

#    def fit(self, encodings):
#        NearestNeighbors(n_neighbors=self.n_neighbors).fit(encodings)
#        return self

#    def predict(self, image):
#        distances, indices = self.kneighbors(images)

#        return distances, indices

       
#if __name__ == '__main__':
#    nbrs = NearestNeighbors()
#    nbrs.fit([1,2,3,4])
