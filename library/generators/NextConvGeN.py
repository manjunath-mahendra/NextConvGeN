import numpy as np
import matplotlib.pyplot as plt

from library.interfaces import GanBaseClass
from library.dataset import DataSet
from library.timing import timing

from keras.layers import Dense, Input, Multiply, Flatten, Conv1D, Reshape
from keras.models import Model
from keras import backend as K
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Lambda

from sklearn.utils import shuffle

from library.NNSearch import NNSearch, randomIndices

import warnings
warnings.filterwarnings("ignore")



def repeat(x, times):
    return [x for _i in range(times)]

def create01Labels(totalSize, sizeFirstHalf):
    labels = repeat(np.array([1,0]), sizeFirstHalf)
    labels.extend(repeat(np.array([0,1]), totalSize - sizeFirstHalf))
    return np.array(labels)

class NextConvGeN(GanBaseClass):
    """
    This is the ConvGeN class. ConvGeN is a synthetic point generator for imbalanced datasets.
    """
    def __init__(self, n_feat, neb=5, gen=None, neb_epochs=10, fdc=None, maj_proximal=False, debug=False, alpha_clip=0):
        self.isTrained = False
        self.n_feat = n_feat
        self.neb = neb
        self.nebInitial = neb
        self.genInitial = gen
        self.gen = gen if gen is not None else self.neb
        self.neb_epochs = neb_epochs
        self.loss_history = None
        self.debug = debug
        self.minSetSize = 0
        self.conv_sample_generator = None
        self.maj_min_discriminator = None
        self.maj_proximal = maj_proximal
        self.cg = None
        self.canPredict = True
        self.fdc = fdc
        self.lastProgress = (-1,-1,-1)
        self.alpha_clip = alpha_clip
        
        self.timing = { n: timing(n) for n in [
            "Train", "BMB", "NbhSearch", "NBH", "GenSamples", "Fit", "FixType"
            ] }
        

        if self.neb is not None and self.gen is not None and self.neb > self.gen:
            raise ValueError(f"Expected neb <= gen but got neb={neb} and gen={gen}.")

    def reset(self, data):
        """
        Creates the network.

        *dataSet* is a instance of /library.dataset.DataSet/ or None.
        It contains the training dataset.
        It is used to determine the neighbourhood size if /neb/ in /__init__/ was None.
        """
        self.isTrained = False

        if data is not None:
            nMinoryPoints = data.shape[0]
            if self.nebInitial is None:
                self.neb = nMinoryPoints
            else:
                self.neb = min(self.nebInitial, nMinoryPoints)
        else:
            self.neb = self.nebInitial

        self.gen = self.genInitial if self.genInitial is not None else self.neb

        ## instanciate generator network and visualize architecture
        self.conv_sample_generator = self._conv_sample_gen()

        ## instanciate discriminator network and visualize architecture
        self.maj_min_discriminator = self._maj_min_disc()

        ## instanciate network and visualize architecture
        self.cg = self._convGeN(self.conv_sample_generator, self.maj_min_discriminator)

        self.lastProgress = (-1,-1,-1)
        if self.debug:
            print(f"neb={self.neb}, gen={self.gen}")

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

        self.timing["Train"].start()
        # Store size of minority class. This is needed during point generation.
        self.minSetSize = data.shape[0]

        normalizedData = data
        if self.fdc is not None:
            normalizedData = self.fdc.normalize(data)
            
        print(f"|N| = {normalizedData.shape}")
        print(f"|D| = {data.shape}")
        
        self.timing["NbhSearch"].start()
        # Precalculate neighborhoods
        self.nmbMin = NNSearch(self.neb).fit(haystack=normalizedData)
        self.nmbMin.basePoints = np.array([ [x.astype(np.float32) for x in p] for p in data])
        self.timing["NbhSearch"].stop()

        # Do the training.
        self._rough_learning(data, discTrainCount, batchSize=batchSize)
        
        # Neighborhood in majority class is no longer needed. So save memory.
        self.isTrained = True
        self.timing["Train"].stop()

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
        runs = (synth_num // self.gen) + 1

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
        batch = neighborhoods.take(runs * self.minSetSize).batch(32)
        

        synth_batch = self.conv_sample_generator.predict(batch)
        
        

        n = 0
        synth_set = []
        for (x,y) in zip(neighborhoods, synth_batch):
            synth_set.extend(self.correct_feature_types(x.numpy(), y))
            n += len(y)
            if n >= numOfSamples:
                break

        ## extract the exact number of synthetic samples needed to exactly balance the two classes
        return np.array(synth_set[:numOfSamples])

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
        tf.random.set_seed(42)
        ## takes minority batch as input
        min_neb_batch = Input(shape=(self.neb, self.n_feat,))

        ## using 1-D convolution, feature dimension remains the same
        x = Conv1D(self.n_feat, 3, activation='relu')(min_neb_batch)
        ## flatten after convolution
        x = Flatten()(x)
        ## add dense layer to transform the vector to a convenient dimension
        x = Dense(self.neb * self.gen, activation='relu')(x)

        ## again, witching to 2-D tensor once we have the convenient shape
        x = Reshape((self.neb, self.gen))(x)
        ## column wise sum
        s = K.sum(x, axis=1)
        ## adding a small constant to always ensure the column sums are non zero.
        ## if this is not done then during initialization the sum can be zero.
        s_non_zero = Lambda(lambda x: x + .000001)(s)
        tf.random.set_seed(42)
        ## reprocals of the approximated column sum
        sinv = tf.math.reciprocal(s_non_zero)
        ## At this step we ensure that column sum is 1 for every row in x.
        ## That means, each column is set of convex co-efficient
        x = Multiply()([sinv, x])
        ## Now we transpose the matrix. So each row is now a set of convex coefficients
        aff=tf.transpose(x[0])
        ## We now do matrix multiplication of the affine combinations with the original
        ## minority batch taken as input. This generates a convex transformation
        ## of the input minority batch
        
        
        @tf.function
        def clipping_alpha(x, clip=self.alpha_clip):
            max_val = tf.math.reduce_max(x, axis=1)
            min_val = tf.math.reduce_min(x, axis=1)
            clip_amt = clip * max_val

            # Create a copy of the input tensor to modify
            x_mod = tf.identity(x)

            for row in range(x.shape[0]):
                max_done = tf.constant(0, dtype=tf.int32)
                min_done = tf.constant(0, dtype=tf.int32)

                for element in range(x.shape[1]):
                    if tf.math.logical_and(
                            tf.math.equal(x[row, element], max_val[row]),
                            tf.math.equal(max_done, 0)):
                        x_mod = tf.tensor_scatter_nd_update(
                            x_mod, [[row, element]], [max_val[row] - clip_amt[row]])
                        max_done = tf.constant(1, dtype=tf.int32)
                    elif tf.math.logical_and(
                            tf.math.equal(x[row, element], min_val[row]),
                            tf.math.equal(min_done, 0)):
                        x_mod = tf.tensor_scatter_nd_update(
                            x_mod, [[row, element]], [min_val[row] + clip_amt[row]])
                        min_done = tf.constant(1, dtype=tf.int32)

            return x_mod


        aff=Lambda(clipping_alpha)(aff)
        
        synth=tf.matmul(aff, min_neb_batch)
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
        samples = Input(shape=(self.n_feat,))
        
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
        ## by default the discriminator trainability is switched off.
        ## Thus training ConvGeN means training the generator network as per previously
        ## trained discriminator network.
        discriminator.trainable = False
        
        # Shape of data:  (batchSize, 2, gen, n_feat)
        # Shape of labels: (batchSize, 2 * gen, 2) 

        ## input receives a neighbourhood minority batch
        ## and a proximal majority batch concatenated
        batch_data = Input(shape=(2, self.gen, self.n_feat,))
        # batch_data: (batchSize, 2, gen, n_feat)
        
        ## extract minority batch
        min_batch = Lambda(lambda x: x[:, 0, : ,:], name="SplitForGen")(batch_data)
        # min_batch: (batchSize, gen, n_feat)
        
        ## extract majority batch
        maj_batch = Lambda(lambda x: x[:, 1, :, :], name="SplitForDisc")(batch_data)
        # maj_batch: (batchSize, gen, n_feat)
        maj_batch = tf.reshape(maj_batch, (-1, self.n_feat), name="ReshapeForDisc")
        # maj_batch: (batchSize * gen, n_feat)
        
        ## pass minority batch into generator to obtain convex space transformation
        ## (synthetic samples) of the minority neighbourhood input batch
        conv_samples = generator(min_batch)
        # conv_batch: (batchSize, gen, n_feat)
        conv_samples = tf.reshape(conv_samples, (-1, self.n_feat), name="ReshapeGenOutput")
        # conv_batch: (batchSize * gen, n_feat)

        ## pass samples into the discriminator to know its decisions
        conv_samples = discriminator(conv_samples)
        conv_samples = tf.reshape(conv_samples, (-1, self.gen, 2), name="ReshapeGenDiscOutput")
        # conv_batch: (batchSize * gen, 2)

        maj_batch = discriminator(maj_batch)
        maj_batch = tf.reshape(maj_batch, (-1, self.gen, 2), name="ReshapeGenDiscOutput")
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
        generator = self.conv_sample_generator
        discriminator = self.maj_min_discriminator
        convGeN = self.cg
        loss_history = [] ## this is for stroring the loss for every run
        minSetSize = len(data)

        ## Create labels for one neighborhood training.
        nLabels = 2 * self.gen
        labels = np.array(create01Labels(nLabels, self.gen))
        labelsGeN = np.array([labels])
        
        def indexToBatches(min_idx):
            self.timing["NBH"].start()
            ## generate minority neighbourhood batch for every minority class sampls by index
            min_batch_indices = self.nmbMin.neighbourhoodOfItem(min_idx)
            min_batch = self.nmbMin.getPointsFromIndices(min_batch_indices)

            ## generate random proximal majority batch
            maj_batch = self._BMB(min_batch_indices)
            self.timing["NBH"].stop()

            return (min_batch, maj_batch)

        def createSamples(min_idx):
            min_batch, maj_batch = indexToBatches(min_idx)

            self.timing["GenSamples"].start()
            ## generate synthetic samples from convex space
            ## of minority neighbourhood batch using generator
            conv_samples = generator.predict(np.array([min_batch]), batch_size=self.neb)
            conv_samples = tf.reshape(conv_samples, shape=(self.gen, self.n_feat))
            self.timing["GenSamples"].stop()

            self.timing["FixType"].start()
            ## Fix feature types
            conv_samples = self.correct_feature_types(min_batch.numpy(), conv_samples)
            self.timing["FixType"].stop()

            ## concatenate them with the majority batch
            conv_samples = [conv_samples, maj_batch]
            return conv_samples

        def genSamplesForDisc():
            for min_idx in range(minSetSize):
                yield createSamples(min_idx)

        def genSamplesForGeN():
            for min_idx in range(minSetSize):
                yield indexToBatches(min_idx)

        def unbatch(rows):
            def fn():
                for row in rows:
                    for part in row:
                        for x in part:
                            yield x
            return fn

        def genLabels():
            for min_idx in range(minSetSize):
                for x in labels:
                    yield x
        
        padd = np.zeros((self.gen - self.neb, self.n_feat))
        discTrainCount = 1 + max(0, discTrainCount)    

        for neb_epoch_count in range(self.neb_epochs):
            self.progressBar([(neb_epoch_count + 1) / self.neb_epochs, 0.5, 0.5])

            ## Training of the discriminator.
            #
            # Get all neighborhoods and synthetic points as data stream.
            a = tf.data.Dataset.from_generator(genSamplesForDisc, output_types=tf.float32).repeat().take(discTrainCount * self.minSetSize)
            a = tf.data.Dataset.from_generator(unbatch(a), output_types=tf.float32)

            # Get all labels as data stream.
            b = tf.data.Dataset.from_tensor_slices(labels).repeat()

            # Zip data and matching labels together for training. 
            samples = tf.data.Dataset.zip((a, b)).batch(batchSize * 2 * self.gen)

            # train the discriminator with the concatenated samples and the one-hot encoded labels
            self.timing["Fit"].start()
            discriminator.trainable = True
            discriminator.fit(x=samples, verbose=0)
            discriminator.trainable = False
            self.timing["Fit"].stop()

            ## use the complete network to make the generator learn on the decisions
            ## made by the previous discriminator training
            #
            # Get all neighborhoods as data stream.
            a = (tf.data.Dataset
                .from_generator(genSamplesForGeN, output_types=tf.float32)
                .map(lambda x: [[tf.concat([x[0], padd], axis=0), x[1]]]))

            # Get all labels as data stream.
            b = tf.data.Dataset.from_tensor_slices(labelsGeN).repeat()

            # Zip data and matching labels together for training. 
            samples = tf.data.Dataset.zip((a, b)).batch(batchSize)

            # Train with the data stream. Store the loss for later usage.
            gen_loss_history = convGeN.fit(samples, verbose=0, batch_size=batchSize)
            loss_history.append(gen_loss_history.history['loss'])


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
        self.timing["BMB"].start()
        indices = randomIndices(self.minSetSize, outputSize=self.gen, indicesToIgnore=min_idxs)
        r = self.nmbMin.basePoints[indices]
        self.timing["BMB"].stop()
        return r


    def retrainDiscriminitor(self, data, labels):
        self.maj_min_discriminator.trainable = True
        labels = np.array([ [x, 1 - x] for x in labels])
        self.maj_min_discriminator.fit(x=data, y=labels, batch_size=20, epochs=self.neb_epochs)
        self.maj_min_discriminator.trainable = False

    def progressBar(self, x):
        x = [int(v * 10) for v in x]
        if True not in [self.lastProgress[i] != x[i] for i in range(len(self.lastProgress))]:
            return
        
        def bar(v):   
            r = ""
            for n in range(10):
                if n > v:
                    r += " "
                else:
                    r += "="
            return r
        
        s = [bar(v) for v in x]
        print(f"[{s[0]}] [{s[1]}] [{s[2]}]", end="\r")
        
    def correct_feature_types(self, batch, synth_batch):
        if self.fdc is None:
            return synth_batch
        
        def bestMatchOf(referenceValues, value):
            if referenceValues is not None:
                best = referenceValues[0]
                d = abs(best - value)
                for x in referenceValues:
                    dx = abs(x - value)
                    if dx < d:
                        best = x
                        d = dx
                return best
            else:
                return value
        
        def correctVector(referenceLists, v):
            return np.array([bestMatchOf(referenceLists[i], v[i]) for i in range(len(v))])
            
        referenceLists = [None for _ in range(self.n_feat)]
        for i in (self.fdc.nom_list or []):
            referenceLists[i] = list(set(list(batch[:, i])))

        for i in (self.fdc.ord_list or []):
            referenceLists[i] = list(set(list(batch[:, i])))

        # print(batch.shape, synth_batch.shape)

        return Lambda(lambda x: np.array([correctVector(referenceLists, y) for y in x]))(synth_batch)
