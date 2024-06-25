import numpy as np
import matplotlib.pyplot as plt

from library.interfaces import GanBaseClass

from keras.layers import Dense, Input, Multiply, Flatten, Conv1D, Reshape, InputLayer, Add
from keras.models import Model, Sequential
from keras import backend as K

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Lambda
import tensorflow_probability as tfp

from library.NNSearch import NNSearch, randomIndices

import warnings
warnings.filterwarnings("ignore")



def repeat(x, times):
    return [x for _i in range(times)]

def create01Labels(totalSize, sizeFirstHalf):
    labels = repeat(np.array([1,0]), sizeFirstHalf)
    labels.extend(repeat(np.array([0,1]), totalSize - sizeFirstHalf))
    return np.array(labels)


class GeneratorConfig:
    def __init__(self, n_feat=None, neb=5, gen=None, neb_epochs=10, genLayerSizes=None, genAddNoise=True, alpha_clip=0):
        self.n_feat = n_feat
        self.neb = neb
        self.gen = gen
        self.neb_epochs = neb_epochs
        self.genAddNoise = genAddNoise
        self.genLayerSizes = genLayerSizes
        self.alpha_clip = alpha_clip

    def isConfigMissing(self):
        return any( x is None for x in
            [ self.n_feat
            , self.neb
            , self.gen
            , self.genAddNoise
            , self.genLayerSizes
            , self.neb_epochs
            ])

    def checkForValidConfig(self):
        if self.isConfigMissing():
            raise ValueError(f"Some configuration is missing.")

        if self.neb > self.gen:
            raise ValueError(f"Expected neb <= gen but got neb={self.neb} and gen={self.gen}.")

        if sum(self.genLayerSizes) != self.gen:
            raise ValueError(f"Expected the layer sizes to sum up to gen={self.gen}.")

        return True

    def fixMissingValuesByInputData(self, data):
        config = GeneratorConfig()
        config.neb = self.neb
        config.gen = self.gen
        config.genAddNoise = self.genAddNoise
        config.genLayerSizes = self.genLayerSizes
        
        if data is not None:
            if config.n_feat is None:
                config.n_feat = data.shape[1]

            if config.neb is None:
                config.neb = data.shape[0]
            else:
                config.neb = min(config.neb, data.shape[0])

        if config.gen is None:
            config.gen = config.neb

        if config.genLayerSizes is None:
            config.genLayerSizes = [config.gen]

        return config

    def nebShape(self, aboveSize=None):
        if aboveSize is None:
            return (self.neb, self.n_feat)
        else:
            return (aboveSize, self.neb, self.n_feat)

    def genShape(self, aboveSize=None):
        if aboveSize is None:
            return (self.gen, self.n_feat)
        else:
            return (aboveSize, self.gen, self.n_feat)



class XConvGeN(GanBaseClass):
    """
    This is the ConvGeN class. ConvGeN is a synthetic point generator for imbalanced datasets.
    """
    def __init__(self, config=None, fdc=None, debug=False):
        self.isTrained = False
        self.config = config
        self.defaultConfig = config
        self.loss_history = None
        self.debug = debug
        self.minSetSize = 0
        self.conv_sample_generator = None
        self.maj_min_discriminator = None
        self.cg = None
        self.canPredict = True
        self.fdc = fdc
        self.lastProgress = -1
        
        self.debugList = []

        if not self.config.isConfigMissing():
            self.config.checkForValidConfig()

    def reset(self, data):
        """
        Creates the network.

        *dataSet* is a instance of /library.dataset.DataSet/ or None.
        It contains the training dataset.
        It is used to determine the neighbourhood size if /neb/ in /__init__/ was None.
        """
        self.isTrained = False

        self.config = self.defaultConfig.fixMissingValuesByInputData(data)
        self.config.checkForValidConfig()

        ## instanciate generator network and visualize architecture
        self.conv_sample_generator = self._conv_sample_gen()

        ## instanciate discriminator network and visualize architecture
        self.maj_min_discriminator = self._maj_min_disc()

        ## instanciate network and visualize architecture
        self.cg = self._convGeN(self.conv_sample_generator, self.maj_min_discriminator)

        self.lastProgress = (-1,-1,-1)
        if self.debug:
            print(f"neb={self.config.neb}, gen={self.config.gen}")

            print(self.conv_sample_generator.summary())
            print('\n')
            
            print(self.maj_min_discriminator.summary())
            print('\n')

            print(self.cg.summary())
            print('\n')

    def train(self, data, discTrainCount=5, batchSize=32):
        """
        Trains the Network.

        *dataSet* is a instance of /library.dataset.DataSet/. It contains the training dataset.
        
        *discTrainCount* gives the number of extra training for the discriminator for each epoch. (>= 0)
        """
        if data.shape[0] <= 0:
            raise AttributeError("Train: Expected data class 1 to contain at least one point.")

        # Store size of minority class. This is needed during point generation.
        self.minSetSize = data.shape[0]

        normalizedData = data
        if self.fdc is not None:
            normalizedData = self.fdc.normalize(data)
            
        # Precalculate neighborhoods
        self.nmbMin = NNSearch(self.config.neb).fit(haystack=normalizedData)
        self.nmbMin.basePoints = np.array([ [x.astype(np.float32) for x in p] for p in data])

        # Do the training.
        self._rough_learning(data, discTrainCount, batchSize=batchSize)
        
        # Neighborhood in majority class is no longer needed. So save memory.
        self.isTrained = True

    def generateDataPoint(self):
        """
        Returns one synthetic data point by repeating the stored list.
        """
        return (self.generateData(1))[0]


    def generateData(self, numOfSamples=1):
        """
        Generates a list of synthetic data-points.

        *numOfSamples* is a integer > 0. It gives the number of new generated samples.
        """
        if not self.isTrained:
            raise ValueError("Try to generate data with untrained network.")

        ## roughly claculate the upper bound of the synthetic samples to be generated from each neighbourhood
        synth_num = (numOfSamples // self.minSetSize) + 1
        runs = (synth_num // self.config.gen) + 1

        ## Get a random list of all indices
        indices = randomIndices(self.minSetSize)

        ## generate all neighborhoods
        def neighborhoodGenerator():
            for index in indices:
                yield self.nmbMin.getNbhPointsOfItem(index)

        neighborhoods = (tf.data.Dataset
            .from_generator(neighborhoodGenerator, output_types=tf.float32)
            .repeat()
            )
        batch = neighborhoods.take(runs * self.minSetSize)

        synth_batch = self.conv_sample_generator.predict(batch.batch(32, deterministic=True), verbose=0)

        pairs = tf.data.Dataset.zip(
            ( batch
            , tf.data.Dataset.from_tensor_slices(synth_batch)
            ))

        corrected = pairs.map(self.correct_feature_types())

        ## extract the exact number of synthetic samples needed to exactly balance the two classes
        r = np.concatenate(np.array(list(corrected.take(1 + (numOfSamples // self.config.gen)))), axis=0)[:numOfSamples]

        return r

    def predictReal(self, data):
        """
        Uses the discriminator on data.
        
        *data* is a numpy array of shape (n, n_feat) where n is the number of datapoints and n_feat the number of features.
        """
        prediction = self.maj_min_discriminator.predict(data)
        return np.array([x[0] for x in prediction])

    # ###############################################################
    # Hidden internal functions
    # ###############################################################

    # Creating the Network: Generator
    def _conv_sample_gen(self):
        """
        The generator network to generate synthetic samples from the convex space
        of arbitrary minority neighbourhoods
        """

        @tf.function
        def clipping_alpha(x, clip=self.config.alpha_clip):
            max_val = tf.math.reduce_max(x, axis=1)
            clip_amt = clip * max_val
       
            # Create a copy of the input tensor to modify
            zp = tf.zeros((x.shape[0],x.shape[1]))
            zm = tf.zeros((x.shape[0],x.shape[1]))
            for row in range(x.shape[0]):
                pos_max = tf.argmax(x[row,:])
                pos_min = tf.argmin(x[row,:])
                c = clip_amt[row]
                zp = tf.tensor_scatter_nd_update(zp, [(row, pos_min)], [c])
                zm = tf.tensor_scatter_nd_update(zm, [(row, pos_max)], [c])
            
            x_mod = x + zp - zm
       
            return x_mod
  
        n_feat = self.config.n_feat
        neb = self.config.neb
        gen = self.config.gen
        genLayerSizes = self.config.genLayerSizes

        ## takes minority batch as input
        min_neb_batch = Input(shape=(neb, n_feat))

        ## using 1-D convolution, feature dimension remains the same
        x = Conv1D(n_feat, 3, activation='relu', name="UnsharpenInput")(min_neb_batch)
        ## flatten after convolution
        x = Flatten(name="InputMatrixToVector")(x)

        synth = []
        n = 0
        if sum(genLayerSizes) < gen:
            genLayerSizes.append(gen)

        for layerSize in genLayerSizes:
            w = min(layerSize, gen - n)
            if w <= 0:
                break
            n += w
    
            ## add dense layer to transform the vector to a convenient dimension
            y = Dense(neb * w, activation='relu', name=f"P{n}_dense")(x)

            ## again, witching to 2-D tensor once we have the convenient shape
            y = Reshape((neb, w), name=f"P{n}_reshape")(y)

            ## column wise sum
            s = K.sum(y, axis=1)

            ## adding a small constant to always ensure the column sums are non zero.
            ## if this is not done then during initialization the sum can be zero.
            s_non_zero = Lambda(lambda x: x + .000001, name=f"P{n}_make_non_zero")(s)

            ## reprocals of the approximated column sum
            sinv = tf.math.reciprocal(s_non_zero, name=f"P{n}_invert")

            ## At this step we ensure that column sum is 1 for every row in x.
            ## That means, each column is set of convex co-efficient
            y = Multiply(name=f"P{n}_normalize")([sinv, y])

            ## Now we transpose the matrix. So each row is now a set of convex coefficients
            aff = tf.transpose(y[0], name=f"P{n}_transpose")

            ## We now do matrix multiplication of the affine combinations with the original
            ## minority batch taken as input. This generates a convex transformation
            ## of the input minority batch
  
            aff = Lambda(clipping_alpha)(aff)

            y = tf.matmul(aff, min_neb_batch, name=f"P{n}_project")
            synth.append(y)

        synth = tf.concat(synth, axis=1, name="collect_planes")

        nOut = gen * n_feat

        if self.config.genAddNoise:
            noiseGenerator = Sequential([
              InputLayer(input_shape=(gen, n_feat)),
              Flatten(),
              Dense(tfp.layers.IndependentNormal.params_size(nOut)),
              tfp.layers.IndependentNormal(nOut)
            ], name="RandomNoise")

            noise = noiseGenerator(synth)
            noise = Reshape((gen, n_feat), name="ReshapeNoise")(noise)
            synth = Add(name="AddNoise")([synth, noise])

        ## finally we compile the generator with an arbitrary minortiy neighbourhood batch
        ## as input and a covex space transformation of the same number of samples as output
        model = Model(inputs=min_neb_batch, outputs=synth)
        opt = Adam(learning_rate=0.001)
        model.compile(loss='mean_squared_logarithmic_error', optimizer=opt)
        return model

    # Creating the Network: discriminator
    def _maj_min_disc(self):
        """
        the discriminator is trained in two phase:
        first phase:  while training ConvGeN the discriminator learns to differentiate synthetic
                      minority samples generated from convex minority data space against
                      the borderline majority samples
        second phase: after the ConvGeN generator learns to create synthetic samples,
                      it can be used to generate synthetic samples to balance the dataset
                      and then rettrain the discriminator with the balanced dataset
        """

        ## takes as input synthetic sample generated as input stacked upon a batch of
        ## borderline majority samples
        samples = Input(shape=(self.config.n_feat,))
        
        ## passed through two dense layers
        y = Dense(250, activation='relu')(samples)
        y = Dense(125, activation='relu')(y)
        y = Dense(75, activation='relu')(y)
        
        ## two output nodes. outputs have to be one-hot coded (see labels variable before)
        output = Dense(2, activation='sigmoid')(y)
        
        ## compile model
        model = Model(inputs=samples, outputs=output)
        opt = Adam(learning_rate=0.0001)
        model.compile(loss='binary_crossentropy', optimizer=opt)
        return model

    # Creating the Network: ConvGeN
    def _convGeN(self, generator, discriminator):
        """
        for joining the generator and the discriminator
        conv_coeff_generator-> generator network instance
        maj_min_discriminator -> discriminator network instance
        """

        n_feat = self.config.n_feat
        neb = self.config.neb
        gen = self.config.gen

        ## by default the discriminator trainability is switched off.
        ## Thus training ConvGeN means training the generator network as per previously
        ## trained discriminator network.
        discriminator.trainable = False

        # Shape of data:  (batchSize, 2, gen, n_feat)
        # Shape of labels: (batchSize, 2 * gen, 2) 

        ## input receives a neighbourhood minority batch
        ## and a proximal majority batch concatenated
        batch_data = Input(shape=(2, gen, n_feat))
        # batch_data: (batchSize, 2, gen, n_feat)
        
        ## extract minority batch
        min_batch = Lambda(lambda x: x[:, 0, : ,:], name="SplitForGen")(batch_data)
        # min_batch: (batchSize, gen, n_feat)
        
        ## extract majority batch
        maj_batch = Lambda(lambda x: x[:, 1, :, :], name="SplitForDisc")(batch_data)
        # maj_batch: (batchSize, gen, n_feat)
        maj_batch = tf.reshape(maj_batch, (-1, n_feat), name="ReshapeForDisc")
        # maj_batch: (batchSize * gen, n_feat)
        
        ## pass minority batch into generator to obtain convex space transformation
        ## (synthetic samples) of the minority neighbourhood input batch
        conv_samples = generator(min_batch)
        # conv_batch: (batchSize, gen, n_feat)
        conv_samples = tf.reshape(conv_samples, (-1, n_feat), name="ReshapeGenOutput")
        # conv_batch: (batchSize * gen, n_feat)

        ## pass samples into the discriminator to know its decisions
        conv_samples = discriminator(conv_samples)
        conv_samples = tf.reshape(conv_samples, (-1, gen, 2), name="ReshapeGenDiscOutput")
        # conv_batch: (batchSize * gen, 2)

        maj_batch = discriminator(maj_batch)
        maj_batch = tf.reshape(maj_batch, (-1, gen, 2), name="ReshapeMajDiscOutput")
        # conv_batch: (batchSize * gen, 2)
        
        ## concatenate the decisions
        output = tf.concat([conv_samples, maj_batch],axis=1)
        # output: (batchSize, 2 * gen, 2)
        
        ## note that, the discriminator will not be traied but will make decisions based
        ## on its previous training while using this function
        model = Model(inputs=batch_data, outputs=output)
        opt = Adam(learning_rate=0.0001)
        model.compile(loss='mse', optimizer=opt)
        return model

    # Training
    def _rough_learning(self, data, discTrainCount, batchSize=32):

        n_feat = self.config.n_feat
        neb = self.config.neb
        gen = self.config.gen

        generator = self.conv_sample_generator
        discriminator = self.maj_min_discriminator
        convGeN = self.cg
        loss_history = [] ## this is for stroring the loss for every run
        minSetSize = len(data)

        ## Create labels for one neighborhood training.
        nLabels = 2 * gen
        labels = np.array(create01Labels(nLabels, gen))
        labelsGeN = np.array([labels])

        def getNeighborhoods():
            for index in range(self.minSetSize):
                yield indexToBatches(index)

        
        def indexToBatches(min_idx):
            ## generate minority neighbourhood batch for every minority class sampls by index
            min_batch_indices = self.nmbMin.neighbourhoodOfItem(min_idx)
            min_batch = self.nmbMin.getPointsFromIndices(min_batch_indices)

            ## generate random proximal majority batch
            maj_batch = self._BMB(min_batch_indices)

            return (min_batch, maj_batch)

        
        def genLabels():
            for min_idx in range(minSetSize):
                for x in labels:
                    yield x

        fnCt = self.correct_feature_types()
        
        def myMysticFunction(xs, ys, zs, ls):
          xs = list(xs.as_numpy_iterator())
          ys = list(ys.as_numpy_iterator())
          zs = list(zs.as_numpy_iterator())

          def g():
            i = 0
            k = 0
            n = len(xs)
            m = len(ls)
            while i < n:
              x = xs[i]
              y = ys[i]
              z = zs[i]
              i += 1

              for nbh in fnCt(x,y):
                 yield (nbh, ls[k])
                 k = (k + 1) % m

              for nbh in z:
                 yield (nbh, ls[k])
                 k = (k + 1) % m
          return g


        def myZip(xs, ys):
            xs = iter(xs)
            ys = iter(ys)
            def g():
                while True:
                    x = next(xs, None)
                    y = next(ys, None)
                    if x is None or y is None:
                        break
                    yield (x,y)
            return g
        
        padd = np.zeros((gen - neb, n_feat))
        discTrainCount = 1 + max(0, discTrainCount)

        for neb_epoch_count in range(self.config.neb_epochs):
            self.progressBar(neb_epoch_count / self.config.neb_epochs)

            ## Training of the discriminator.
            #
            # Get all neighborhoods and synthetic points as data stream.
            nbhPairs = tf.data.Dataset.from_generator(getNeighborhoods, output_types=tf.float32).repeat().take(discTrainCount * self.minSetSize)
            nbhMin = nbhPairs.map(lambda x: x[0], deterministic=True)
            batchMaj = nbhPairs.map(lambda x: x[1], deterministic=True)

            batch_nbhMin = nbhMin.batch(32, deterministic=True)
            synth_batch = self.conv_sample_generator.predict(batch_nbhMin , verbose=0)
            fnGen = myMysticFunction(nbhMin, tf.data.Dataset.from_tensor_slices(synth_batch), batchMaj, labels)
            samples = tf.data.Dataset.from_generator(fnGen, output_signature=(tf.TensorSpec(shape=(n_feat,), dtype=tf.int64, name=None), tf.TensorSpec(shape=(2,), dtype=tf.int64, name=None)))          
            samples = samples.batch(batchSize * 2 * gen, deterministic=True)

            # train the discriminator with the concatenated samples and the one-hot encoded labels
            discriminator.trainable = True
            discriminator.fit(x=samples, verbose=0, shuffle=False)
            discriminator.trainable = False

            ## use the complete network to make the generator learn on the decisions
            ## made by the previous discriminator training
            #
            # Get all neighborhoods as data stream.
            a = (tf.data.Dataset
                .from_generator(getNeighborhoods, output_types=tf.float32)
                .map(lambda x: [[tf.concat([x[0], padd], axis=0), x[1]]], deterministic=True))

            # Get all labels as data stream.
            b = tf.data.Dataset.from_tensor_slices(labelsGeN).repeat()

            # Zip data and matching labels together for training. 
            samples = tf.data.Dataset.zip((a, b)).batch(batchSize, deterministic=True)

            # Train with the data stream. Store the loss for later usage.
            gen_loss_history = convGeN.fit(samples, verbose=0, batch_size=batchSize, shuffle=False)
            loss_history.append(gen_loss_history.history['loss'])

        self.progressBar(1.0)

        ## When done: print some statistics.
        if self.debug:
            run_range = range(1, len(loss_history) + 1)
            plt.rcParams["figure.figsize"] = (16,10)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.xlabel('runs', fontsize=25)
            plt.ylabel('loss', fontsize=25)
            plt.title('Rough learning loss for discriminator', fontsize=25)
            plt.plot(run_range, loss_history)
            plt.show()

        ## When done: print some statistics.
        self.loss_history = loss_history


    def _BMB(self, min_idxs):

        ## Generate a borderline majority batch
        ## data_maj -> majority class data
        ## min_idxs -> indices of points in minority class
        ## gen -> convex combinations generated from each neighbourhood
        indices = randomIndices(self.minSetSize, outputSize=self.config.gen, indicesToIgnore=min_idxs)
        #self.debugList.append(indices[0])
        r = self.nmbMin.basePoints[indices]
        return r


    def retrainDiscriminitor(self, data, labels):
        self.maj_min_discriminator.trainable = True
        labels = np.array([ [x, 1 - x] for x in labels])
        self.maj_min_discriminator.fit(x=data, y=labels, batch_size=20, epochs=self.config.neb_epochs)
        self.maj_min_discriminator.trainable = False

    def progressBar(self, x):
        barWidth = 40

        x = int(x * barWidth)
        if self.lastProgress == x:
            return
        
        def bar(v):   
            v = min(v, barWidth)
            r = ("=" * v) + (" " * (barWidth - v))
            return r
        
        print(f"[{bar(x)}]", end="\r")
        
    def correct_feature_types(self):
        # batch[0] = original points (gen x n_feat)
        # batch[1] = synthetic points (gen x n_feat)
        
        @tf.function
        def voidFunction(reference, synth):
            return synth
    
        if self.fdc is None:
            return voidFunction
        
        columns = set(self.fdc.nom_list or [])
        for y in (self.fdc.ord_list or []):
            columns.add(y)
        columns = list(columns)
        
        if len(columns) == 0:
            return voidFunction
        
        neb = self.config.neb
        n_feat = self.config.n_feat
        nn = tf.constant([(1.0 if x in columns else 0.0) for x in range(n_feat)])
        if n_feat is None:
            print("ERRROR n_feat is None")

        if nn is None:
            print("ERRROR nn is None")

        @tf.function
        def bestMatchOf(vi):
            value = vi[0]
            c = vi[1][0]
            r = vi[2]
            if c != 0.0:
                d = tf.abs(value - r)
                return r[tf.math.argmin(d)]
            else:
                return value[0]
            
        @tf.function
        def indexted(v, rt):
            vv = tf.reshape(tf.repeat([v], neb, axis=1), (n_feat, neb))
            vn = tf.reshape(tf.repeat([nn], neb, axis=1), (n_feat, neb))
            return tf.stack((vv, vn, rt), axis=1)
        
        @tf.function
        def correctVector(v, rt):
            return tf.map_fn(lambda x: bestMatchOf(x), indexted(v, rt))

        @tf.function
        def fn(reference, synth):
            rt = tf.transpose(reference)
            return tf.map_fn(lambda x: correctVector(x, rt), synth)
        
        return fn
