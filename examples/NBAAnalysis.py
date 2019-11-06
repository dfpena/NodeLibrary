
#!/usr/bin/env python
from decorator import decorator

import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import tensorflow as tf
import ScrapeNBANode


@decorator
def on_start(func,*args, **kwargs):
    if kwargs !={}:
        try:
            if kwargs['Start']:
                if 'Verbose' in kwargs['Settings']:
                    if kwargs['Settings']['Verbose']:
                        print(func)
                        pass
                response= func(*args,**kwargs)
                return response
            else:
                kwargs['Start'] = False
                print(func,"DID NOT START")
                return(kwargs)
        except Exception as e:
            print('NODE ERROR OCCURED TRYING TO START NODE FUNCTION:')
            print('===========================================')
            print(func,e)
            print('===========================================')
            print('LAST STATE SET TO:')
            print('===========================================')
            print('ekwargs')
            print('===========================================')
            print('LAST NODE FUNCTION SET TO:')
            print('===========================================')
            print('efunc')
            print('===========================================')
            global ekwargs
            global efunc
            ekwargs = kwargs
            efunc = func
            print('HALTING')
            raise
    else:
        print('Empty kwargs')
        return ()



def start():
    return {'Start':True,'Settings':{'Verbose':True},'Status':{}}

 
@on_start
def NBANode1(*args,**kwargs):
    nkwargs = kwargs.copy()
    nkwargs['Data'] = 'https://www.basketball-reference.com/players/b/bryanko01.html'
    node = ScrapeNBANode.StremeNode()
    kwargs['Data'] = node.run(**nkwargs)
    kwargs['Data']['Player'] = 'Kobe'
    
    return kwargs
 
@on_start
def reFormat4Som2(*args,**kwargs):
    kwargs['Data']['subject'] = kwargs['Data']['Player'] + kwargs['Data']['Age']
    kwargs['Data'] = kwargs['Data'][['subject','PER','WS','TOV%']]
    return kwargs
 
class SOM(object):
    """
    2-D Self-Organizing Map with Gaussian Neighbourhood function
    and linearly decreasing learning rate.
    """
 
    #To check if the SOM has been trained
    _trained = False
 
    def __init__(self, m, n, dim, n_iterations=100, alpha=None, sigma=None):
        """
        Initializes all necessary components of the TensorFlow
        Graph.
 
        m X n are the dimensions of the SOM. 'n_iterations' should
        should be an integer denoting the number of iterations undergone
        while training.
        'dim' is the dimensionality of the training inputs.
        'alpha' is a number denoting the initial time(iteration no)-based
        learning rate. Default value is 0.3
        'sigma' is the the initial neighbourhood value, denoting
        the radius of influence of the BMU while training. By default, its
        taken to be half of max(m, n).
        """
 
        #Assign required variables first
        self._m = m
        self._n = n
        if alpha is None:
            alpha = 0.3
        else:
            alpha = float(alpha)
        if sigma is None:
            sigma = max(m, n) / 2.0
        else:
            sigma = float(sigma)
        self._n_iterations = abs(int(n_iterations))
 
        ##INITIALIZE GRAPH
        self._graph = tf.Graph()
 
        ##POPULATE GRAPH WITH NECESSARY COMPONENTS
        with self._graph.as_default():
 
            ##VARIABLES AND CONSTANT OPS FOR DATA STORAGE
 
            #Randomly initialized weightage vectors for all neurons,
            #stored together as a matrix Variable of size [m*n, dim]
            self._weightage_vects = tf.Variable(tf.random_normal(
                [m*n, dim]))
 
            #Matrix of size [m*n, 2] for SOM grid locations
            #of neurons
            self._location_vects = tf.constant(np.array(
                list(self._neuron_locations(m, n))))
 
            ##PLACEHOLDERS FOR TRAINING INPUTS
            #We need to assign them as attributes to self, since they
            #will be fed in during training
 
            #The training vector
            self._vect_input = tf.placeholder("float", [dim])
            #Iteration number
            self._iter_input = tf.placeholder("float")
 
            ##CONSTRUCT TRAINING OP PIECE BY PIECE
            #Only the final, 'root' training op needs to be assigned as
            #an attribute to self, since all the rest will be executed
            #automatically during training
 
            #To compute the Best Matching Unit given a vector
            #Basically calculates the Euclidean distance between every
            #neuron's weightage vector and the input, and returns the
            #index of the neuron which gives the least value
            bmu_index = tf.argmin(tf.sqrt(tf.reduce_sum(
                tf.pow(tf.subtract(self._weightage_vects, tf.stack(
                    [self._vect_input for i in range(m*n)])), 2), 1)),
                                  0)
 
            #This will extract the location of the BMU based on the BMU's
            #index
            slice_input = tf.pad(tf.reshape(bmu_index, [1]),
                                 np.array([[0, 1]]))
            bmu_loc = tf.reshape(tf.slice(self._location_vects, slice_input,
                                          tf.constant(np.array([1, 2]))),
                                 [2])
 
            #To compute the alpha and sigma values based on iteration
            #number
            learning_rate_op = tf.subtract(1.0, tf.math.divide(self._iter_input,
                                                  self._n_iterations))
            _alpha_op = tf.multiply(alpha, learning_rate_op)
            _sigma_op = tf.multiply(sigma, learning_rate_op)
 
            #Construct the op that will generate a vector with learning
            #rates for all neurons, based on iteration number and location
            #wrt BMU.
            bmu_distance_squares = tf.reduce_sum(tf.pow(tf.subtract(
                self._location_vects, tf.stack(
                    [bmu_loc for i in range(m*n)])), 2), 1)
            neighbourhood_func = tf.exp(tf.negative(tf.math.divide(tf.cast(
                bmu_distance_squares, "float32"), tf.pow(_sigma_op, 2))))
            learning_rate_op = tf.multiply(_alpha_op, neighbourhood_func)
 
            #Finally, the op that will use learning_rate_op to update
            #the weightage vectors of all neurons based on a particular
            #input
            learning_rate_multiplier = tf.reshape(learning_rate_op, [self._m * self._n, 1])
            weightage_delta = tf.multiply(
                learning_rate_multiplier,
                tf.subtract(tf.stack([self._vect_input for i in range(m*n)]),
                       self._weightage_vects))                                         
            new_weightages_op = tf.add(self._weightage_vects,
                                       weightage_delta)
            self._training_op = tf.assign(self._weightage_vects,
                                          new_weightages_op)                                       
 
            ##INITIALIZE SESSION
            self._sess = tf.Session()
 
            ##INITIALIZE VARIABLES
            init_op = tf.global_variables_initializer()
            self._sess.run(init_op)
 
    def _neuron_locations(self, m, n):
        """
        Yields one by one the 2-D locations of the individual neurons
        in the SOM.
        """
        #Nested iterations over both dimensions
        #to generate all 2-D locations in the map
        for i in range(m):
            for j in range(n):
                yield np.array([i, j])
 
    def train(self, input_vects):
        """
        Trains the SOM.
        'input_vects' should be an iterable of 1-D NumPy arrays with
        dimensionality as provided during initialization of this SOM.
        Current weightage vectors for all neurons(initially random) are
        taken as starting conditions for training.
        """
 
        #Training iterations
        for iter_no in range(self._n_iterations):
            #Train with each vector one by one
            for input_vect in input_vects:
                self._sess.run(self._training_op,
                               feed_dict={self._vect_input: input_vect,
                                          self._iter_input: iter_no})
 
        #Store a centroid grid for easy retrieval later on
        centroid_grid = [[] for i in range(self._m)]
        self._weightages = list(self._sess.run(self._weightage_vects))
        self._locations = list(self._sess.run(self._location_vects))
        for i, loc in enumerate(self._locations):
            centroid_grid[loc[0]].append(self._weightages[i])
        self._centroid_grid = centroid_grid
 
        self._trained = True
 
    def get_centroids(self):
        """
        Returns a list of 'm' lists, with each inner list containing
        the 'n' corresponding centroid locations as 1-D NumPy arrays.
        """
        if not self._trained:
            raise ValueError("SOM not trained yet")
        return self._centroid_grid
 
    def map_vects(self, input_vects):
        """
        Maps each input vector to the relevant neuron in the SOM
        grid.
        'input_vects' should be an iterable of 1-D NumPy arrays with
        dimensionality as provided during initialization of this SOM.
        Returns a list of 1-D NumPy arrays containing (row, column)
        info for each input vector(in the same order), corresponding
        to mapped neuron.
        """
 
        if not self._trained:
            raise ValueError("SOM not trained yet")
 
        to_return = []
        for vect in input_vects:
            min_index = min([i for i in range(len(self._weightages))],
                            key=lambda x: np.linalg.norm(vect-
                                                         self._weightages[x]))
            to_return.append(self._locations[min_index])
 
        return to_return


@on_start
def SimpleSOM(*args, **kwargs):
    df = kwargs['Data']
    kwargs['Settings']['SOM']={'Data':df}
    params = df.keys()[df.keys()!='subject']
    testdata = df[params].values
    subjects = df.subject
    # Train a 20x30 SOM with 400 iterations
    som = SOM(20, 30, len(params), 400)
    kwargs['Data']=som
    som.train(testdata)

    # Get output grid
    #kwargs['Data']['Centroids'] = som.get_centroids()
    # Map colors to their closest neurons
    kwargs['Settings']['SOM']['Mapped']=som.map_vects(testdata)

    return kwargs
      
 
@on_start
def addPlot4(*args,**kwargs):
    
    matplotlib.rcParams['xtick.major.size'] = 0
    matplotlib.rcParams['ytick.major.size'] = 0
    matplotlib.rcParams['xtick.labelsize'] = 0
    matplotlib.rcParams['ytick.labelsize'] = 0
    # Get output grid
    centroids = kwargs['Data'].get_centroids()
    subjects = kwargs['Settings']['SOM']['Data'].subject
    

    if 'Figures' not in kwargs['Settings']:
        kwargs['Settings']['Figures'] ={}
    
    kwargs['Settings']['Figures']['Self Organizing Map'] = plt.figure('Self Organizing Map')
    plt.imshow(centroids)
    for i, m in enumerate(kwargs['Settings']['SOM']['Mapped']):
        plt.text(
            m[1],
            m[0],
            subjects[i],
            ha='center',
            va='center',
            bbox=dict(
                facecolor='white',
                alpha=0.5,
                lw=0))
    plt.show() # Remove this line if you do not want your program to wait for this figure to render
    
    return kwargs

 


class StremeNode:
    def __init__(self):
        pass

    def run(self,*args,**kwargs):
        self.kwargs=addPlot4(**SimpleSOM(**reFormat4Som2(**NBANode1(**kwargs))))
        return (self.kwargs)

class liveprocess:
    def __init__(self):
        self.status="pending"
    def run(self,expname):
        self.response=addPlot4(**SimpleSOM(**reFormat4Som2(**NBANode1(**start()))))
        self.status="completed"
        return(self.status)

if __name__ == '__main__':
    process = liveprocess()
    process.run('Local')
    