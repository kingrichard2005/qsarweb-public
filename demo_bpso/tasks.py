#-------------------------------------------------------------------------------
# Name:        cs512-a3
# Description: Multiple Linear Regression with Binary Particle Swarm Optimization
#   
# Author:      kingrichard2005
#
# Created:     2014-10-27
# Copyright:   (c) kingrichard2005 2014
# Licence:     MIT
#-------------------------------------------------------------------------------
from __future__ import absolute_import
import argparse
import numpy as np
import hashlib
import math
import sys
import os
import csv

from random import *
from numpy import *
from sklearn import linear_model                    # provides Multiple / Linear Regressor class
from sklearn import svm                             # provides Support Vector Regressor class
from sklearn import neighbors                       # provides K-Nearest Neighbors Regressor class
from sklearn import ensemble                        # provides a Random Forest Regressor class

#Celery Task Queue Imports
from celery.result import allow_join_result
from celery import shared_task, Task
from celery.utils.log import get_task_logger
from django.core.cache import cache
import pika
import time
# serialize to json where necessary, 
#need for serialization of complex data structures
#for message passing.
import json

#todo make serializable

#globals
globalbestposition_vector = []
globalbestfitness         = -1

# helpers

def getLogTime():
    return "{0}  ".format(time.strftime("%H:%M:%S", time.localtime()))

def publishLogMessage(queueId = "default", msg = ""):
    '''Publish a message to the specified queue'''
    try:
        connection   = pika.BlockingConnection();
        channel      = connection.channel()
        # create a message queue identifiable by session id to publish (log) messages
        # to be read by subscribing clients.
        strSessionId = str( queueId )
        channel.queue_declare( queue = strSessionId )
        channel.basic_publish( exchange = '', routing_key = strSessionId, body = msg )
        connection.close();
        return 0;
    except:
        print "error in queueLogger( ... )"
        return -1;

#TODO: Move these helpers to a separate task application
@shared_task
def createAnOutputFile(file_name):
    '''
        Distributed task creates the result output file 
        to be referenced by the calling consumer(s).
        Return the file name for reference by subsequent callers
    '''
    try:
        with file(file_name, 'wb') as fileOut:
            fileW    = csv.writer(fileOut)
            fileW.writerow(['Descriptor ID', 'No. Descriptors', 'Generation', 'Fitness', 'Model','R2', 'Q2', \
                    'R2Pred_Validation', 'R2Pred_Test','SEE_Train', 'SDEP_Validation', 'SDEP_Test', \
                    'y_Train', 'yHat_Train', 'yHat_CV', 'y_validation', 'yHat_validation','y_Test', 'yHat_Test'])
        output = json.dumps( file_name )
        return output
    except:
        print "error in createAnOutputFile"
        return -1

@shared_task
def placeDataIntoArray(fileName):
    '''
        Ditributed task reads csv input to a numpy data array
        and returns it as a json serialized numpy array dtype = int64
    '''
    try:
        # check if this is a valid file
        # TODO: check if this is a well-formatted csv file.
        if os.path.isfile(fileName) == False:
            print 'this is not a valid file'
            return -1;
        else:
            with open(fileName, mode='rbU') as csvfile:
                datareader = csv.reader(csvfile, delimiter=',', quotechar=' ')
                dataArray = array([row for row in datareader], dtype=float64, order='C')

            # we serialize output to json data exchange format usedbetween Celery tasks
            if (min(dataArray.shape) == 1): # flatten arrays of one row or column
                output = json.dumps( dataArray.tolist() )
                return output
            else:
                output = json.dumps( dataArray.tolist() )
                return output
    except:
        print "error in placeDataIntoArray( ... )"
        return -1

def rescaleTheData(TrainX, ValidateX, TestX):
    try:
        # 1 degree of freedom means (ddof) N-1 unbiased estimation
        # TODO: add verbosity flag for this output
        print "reference: shape of train X: {0}, ValidateX: {1}, TestX: {2}".format( TrainX.shape, ValidateX.shape, TestX.shape  )
        TrainXVar  = TrainX.var(axis = 0, ddof=1)
        TrainXMean = TrainX.mean(axis = 0)

        for i in range(0, TrainX.shape[0]):
            TrainX[i,:] = (TrainX[i,:] - TrainXMean)/sqrt(TrainXVar)
        for i in range(0, ValidateX.shape[0]):
            ValidateX[i,:] = (ValidateX[i,:] - TrainXMean)/sqrt(TrainXVar)
        for i in range(0, TestX.shape[0]):
            TestX[i,:] = (TestX[i,:] - TrainXMean)/sqrt(TrainXVar)

        return TrainX, ValidateX, TestX
    except:
        print "error in rescaleTheData( ... )"

# end helpers

# core ML model trainer func
def r2(y, yHat):
    """
      Compute the coefficient of determination
      summarizes the explanatory power of the regression 
      model and is computed from the sums-of-squares terms
      see: http://www.saedsayad.com/mlr.htm
    """
    try:
        numer = ((y - yHat)**2).sum()# Residual Sum of Squares
        denom = ((y - y.mean())**2).sum()
        r2    = 1 - numer/denom
        return r2
    except:
        print "error calculating coefficient of determination"

def r2Pred(yTrain, yTest, yHatTest):
    try:
        numer  = ((yHatTest - yTest)**2).sum()
        denom  = ((yTest - yTrain.mean())**2).sum()
        r2Pred = 1 - numer/denom
        return r2Pred
    except:
        print "error in r2Pred"

def see(p, y, yHat):
    """
    Standard error of estimate
    (Root mean square error)
    """
    try:
        n     = y.shape[0]
        numer = ((y - yHat)**2).sum()
        denom = n - p - 1
        if (denom == 0):
            s = 0
        elif ( (numer/denom) <0 ):
            s = 0.001
        else:
            s = (numer/denom)** 0.5

        return s
    except:
        print "error computing see"

def sdep(y, yHat):
    """
    Standard deviation of error of prediction
    (Root mean square error of prediction)
    """
    try:
        n     = y.shape[0]
        numer = ((y - yHat)**2).sum()
        sdep  = (numer/n)**0.5
        return sdep
    except:
        print "error in sdep"

def rmse(X, Y):
	"""
	Calculate the root-mean-square error (RMSE) also known as root mean
	square deviation (RMSD).
	
	Parameters
	----------
	X : array_like -- Assumed to be 1D.
	Y : array_like -- Assumed to be the same shape as X.
	
	Returns
	-------
	out : float64
	"""

	X = asarray(X, dtype=float64)
	Y = asarray(Y, dtype=float64)
	
	return (sum((X-Y)**2)/len(X))**.5

def training_prediction(set_x, set_y, model):
    """Predict with training set."""
    try:
        yhat = empty_like(set_y)
        for idx in range(0, yhat.shape[0]):
            train_x   = delete(set_x, idx, axis=0)
            train_y   = delete(set_y, idx, axis=0)
            model     = model.fit(train_x, train_y)
            yhat[idx] = model.predict(set_x[idx])
        return yhat
    except:
        print "error in cv_predict"

def calc_fitness(xi, Y, Yhat, c=2):
    """
    Calculate fitness of a prediction.
    
    Parameters
    ----------
    xi : array_like -- Mask of features to measure fitness of. Must be of dtype bool.
    model : object  -- Object to make predictions, usually a regression model object.
    c : float       -- Adjustment parameter.
    
    Returns
    -------
    out: float -- Fitness for the given data.
    
    """
    p       = sum(xi)   # Number of selected parameters
    n       = len(Y)    # Sample size
    numer   = ((Y - Yhat)**2).sum() / n   # Mean square error
    pcn     = p * (c/n)
    if pcn >= 1:
        return 1000
    denom = (1 - pcn)**2
    theFitness = numer/denom
    return theFitness

def InitializeTracks():
    try:
        trackDesc             = {}
        trackIdx              = {}
        trackGen              = {}
        trackFitness          = {}
        trackModel            = {}
        trackR2               = {}
        trackQ2               = {}
        trackR2PredValidation = {}
        trackR2PredTest       = {}
        trackSEETrain         = {}
        trackSDEPValidation   = {}
        trackSDEPTest         = {}
        return  trackDesc, trackIdx, trackGen, trackFitness, trackModel, trackR2, trackQ2, \
                trackR2PredValidation, trackR2PredTest, trackSEETrain, \
                trackSDEPValidation, trackSDEPTest
    except:
        print "error in InitializeTracks"

def initializeYDimension():
    try:
        yTrain         = {}
        yHatTrain      = {}
        yHatCV         = {}
        yValidation    = {}
        yHatValidation = {}
        yTest          = {}
        yHatTest       = {}
        return yTrain, yHatTrain, yHatCV, yValidation, yHatValidation, yTest, yHatTest
    except:
        print "error in initializeYDimension"

def train_ml_model(model, outputfilepath, swarmpopulation, TrainX, TrainY, ValidateX, ValidateY, TestX, TestY, unfit  = 1000, generationLabel = -1):
    try:
        '''Generates the regression model used as a base for genetic algorithm evolution'''
        numOfPop   = len(swarmpopulation);
        fitness    = zeros(numOfPop)
        c          = 2
        false      = 0
        true       = 1
        predictive = false
        
        # model record keeping
        trackDesc, trackIdx, trackGen, trackFitness,trackModel,trackR2, trackQ2, \
        trackR2PredValidation, trackR2PredTest, trackSEETrain, \
        trackSDEPValidation, trackSDEPTest = InitializeTracks()

        yTrain, yHatTrain, yHatCV, yValidation, \
        yHatValidation, yTest, yHatTest = initializeYDimension()

        itFits = 1
        for i in range(numOfPop):
            # Get the indices of non-zero elements of this Particle's
            # current position
            
            # decode serialized swarm particle
            jsonObj          = json.loads(swarmpopulation[i])
            particleInstance = json.loads('{"__type__": "Particle", "currentposition": ' + str(jsonObj['currentposition']) + ', "localbest": ' + str(jsonObj['localbest']) + ', "velocity": ' + str(jsonObj['velocity']) + '}', object_hook=particle_decoder)
            positionVector   = np.array( [e for e in particleInstance.currentposition] )

            xi  =  positionVector.nonzero()[0].tolist();
            idx = hashlib.sha1(array(xi)).digest() # Hash
            if idx in trackFitness.keys():
                # don't recalculate everything if the model has already been validated
                fitness[i] = trackFitness[idx]
                continue
        
            X_train_masked      = TrainX.T[xi].T
            X_validation_masked = ValidateX.T[xi].T
            X_test_masked       = TestX.T[xi].T
            
            if X_train_masked.shape[1] > 0:
               try:
                   # Using MLR implementation from scikit learn
                   model_desc = model.fit(X_train_masked, TrainY)
               except:
                   return unfit, fitness
            else:
               # print "{0}No optimal descriptors at Particle '{1}'s current position vector, skipping...".format( getLogTime(), i );
               # add ridiculously bad fitness value for this particle since it can't be trained
               fitness[i] = 1000;
               continue;
        
            # Predict using trained model
            Yhat_train         = training_prediction(X_train_masked, TrainY, model)
            Yhat_validation    = model.predict(X_validation_masked)
            Yhat_test          = model.predict(X_test_masked)
            
            # Compute R2 statistics (Prediction for Valiation and Test set)
            q2_loo            = r2(TrainY, Yhat_train)
            r2pred_validation = r2Pred(TrainY, ValidateY, Yhat_validation)
            r2pred_test       = r2Pred(TrainY, TestY, Yhat_test)
            Y_fitness         = append(TrainY, ValidateY)
            Yhat_fitness      = append(Yhat_train, Yhat_validation)
            fitness[i]        = calc_fitness(xi, Y_fitness, Yhat_fitness, c)

            if predictive and ((q2_loo < 0.5) or (r2pred_validation < 0.5) or (r2pred_test < 0.5)):
                # if it's not worth recording, just return the fitness
                print "{0}ending the program because predictive value : {1} is not worth recording, so returning the fitness.".format(getLogTime(), predictive);
                continue
            
            # Compute predicted Y_hat for training set.
            Yhat_train      = model.predict(X_train_masked)
            r2_train        = r2(TrainY, Yhat_train)
   
            # Standard error of estimate
            s               = see(X_train_masked.shape[1], TrainY, Yhat_train)
            sdep_validation = sdep(ValidateY, Yhat_validation)
            sdep_test       = sdep(TrainY, Yhat_train)
            idxLength       = len(xi)

            # store stats
            trackDesc[idx]             = str(xi)
            trackIdx[idx]              = idxLength
            trackGen[idx]              = generationLabel
            trackFitness[idx]          = fitness[i]
            trackModel[idx]            = model_desc
            trackR2[idx]               = r2_train
            trackQ2[idx]               = q2_loo
            trackR2PredValidation[idx] = r2pred_validation
            trackR2PredTest[idx]       = r2pred_test
            trackSEETrain[idx]         = s
            trackSDEPValidation[idx]   = sdep_validation
            trackSDEPTest[idx]         = sdep_test
            yTrain[idx]                = TrainY.flatten('C').tolist()
            yHatTrain[idx]             = Yhat_train.flatten('C').tolist()
            yHatCV[idx]                = Yhat_train.flatten('C').tolist()
            yValidation[idx]           = ValidateY.flatten('C').tolist()
            yHatValidation[idx]        = Yhat_validation.flatten('C').tolist()
            yTest[idx]                 = TestY.flatten('C').tolist()
            yHatTest[idx]              = Yhat_test.flatten('C').tolist()
        
        # Write output to file
        write(outputfilepath, trackDesc, trackIdx, trackGen, trackFitness, trackModel, trackR2,\
                    trackQ2,trackR2PredValidation, trackR2PredTest, trackSEETrain, \
                    trackSDEPValidation,trackSDEPTest,yTrain, yHatTrain, yHatCV, \
                    yValidation, yHatValidation, yTest, yHatTest)
        
        return itFits, fitness, Y_fitness, Yhat_fitness, c
    except:
        print "error in train_ml_model( ... )"

def write(outputFilePath, trackDesc, trackIdx, trackGen, trackFitness, trackModel, trackR2, \
          trackQ2,trackR2PredValidation, trackR2PredTest, trackSEETrain, \
          trackSDEPValidation,trackSDEPTest,yTrain, yHatTrain, yHatCV, \
          yValidation, yHatValidation, yTest, yHatTest):
    '''Write output to file'''
    try:
        with file(outputFilePath, 'ab+') as fileOut:
            fileW     = csv.writer(fileOut)
            for key in trackFitness.keys():
                col1  = trackDesc[key];
                col2  = trackIdx[key];
                col3  = trackGen[key];
                col4  = trackFitness[key];
                col5  = trackModel[key]
                col6  = trackR2[key];
                col7  = trackQ2[key];
                col8  = trackR2PredValidation[key];
                col9  = trackR2PredTest[key];
                col10 = trackSEETrain[key];
                col11 = trackSDEPValidation[key];
                col12 = trackSDEPTest[key];
                col13 = yTrain[key];
                col14 = yHatTrain[key];
                col15 = yHatCV[key];
                col16 = yValidation[key];
                col17 = yHatValidation[key];
                col18 = yTest[key];
                col19 = yHatTest[key];
                fileW.writerow([col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12, col13, col14, col15, col16, col17, col18, col19])
        return 0;
    except:
        print "error in write( ... )"
        return -1;

# end core ML model trainer func

# main bpso

# Particle class
class Particle:
    '''
        Class definition for BPSO swarm Particle
        Note: for distributed Celery tasks, this will need to be 'JSON' serializable,
        add necessary serialization behavior
    '''
    def __init__(self, currentposition = [], localbest = [], velocity = []):
        self.currentposition = [i for i in currentposition];
        self.localbest       = [i for i in localbest];
        self.velocity        = [i for i in velocity];
    
    def __str__(self):
        #TODO: complete string output
        return "Particle string"

    def to_JSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    def updatecurrentposition( self, bpso_lambda = 0.01, bpso_alpha = 0.5, bpso_beta = 0.05):
        '''
        A caveat to the BPSO rules in Eq. 7 is the algorithm
        was found to exhibit local optima convergence issues; to 
        overcome this, Shen et al. had 10% of the particles search 
        the space randomly [18]. Instead of a random particle search, 
        we modified the set of rules to incorporate a positional bit 
        mutation factor:    
        see Eq. 12: https://drive.google.com/#folders/0Bzwbc8cqElSAQUU4QXVTb3Q2TTA
        '''
        try:
            # rule set for updating velocity
            global globalbestposition_vector
            tmpNewPosition = np.array( [ e for e in self.currentposition] )
            refVelocity    = np.array( [ e for e in self.velocity] )
            refLocalBest   = np.array( [ e for e in self.localbest] )
            for i in range(  tmpNewPosition.shape[0]  ):
                if 0 < refVelocity[i] <= bpso_alpha:
                    tmpNewPosition[i] = tmpNewPosition[i];
                elif bpso_alpha < refVelocity[0] <= ( .5 * (1 + bpso_alpha) ):
                    tmpNewPosition[i] = refLocalBest[i];
                elif ( .5 * (1 + bpso_alpha) )  < refVelocity[0] <= ( 1 - bpso_beta ):
                    tmpNewPosition[i] = globalbestposition_vector[i];
                elif ( 1 - bpso_beta ) < refVelocity[0] <= 1:
                   tmpNewPosition[i] = ( 1 - tmpNewPosition[i] );
                else:
                    # noop for now if none of the rules above apply...
                    pass;

            self.currentposition = [ e for e in tmpNewPosition];
            return;
        except:
            print "error in Particle.updatecurrentposition(...)"

    def updatelocalbestposition( self, newLocalBestPosition ):
        '''
             Parameters 'cl' and 'c2' represent the acceleration 
             constants which govern the extent to which the
             particles are drawn towards the local and global best positions.
             see: https://drive.google.com/file/d/0Bzwbc8cqElSAWEx3LVFJZmc4Ymc/view?usp=sharing
          '''
        try:
            self.localbest = [e for e in newLocalBestPosition];
        except:
            print "error in Particle.updatelocalbestposition(...)";
   
    def updatevelocity( self, c1 = .75, c2 = .75 ):
       try:
         '''
            Parameters 'cl' and 'c2' represent the acceleration 
            constants which govern the extent to which the
            particles are drawn towards the local and global best positions.
            see: https://drive.google.com/file/d/0Bzwbc8cqElSAWEx3LVFJZmc4Ymc/view?usp=sharing
         '''
         # equation components
         global globalbestposition_vector
         #tmp = []
         #rand1 = random.random();
         #rand2 = random.random();
         #for i in range( self.velocity.shape[0] ):
         #   newVelComponent = self.velocity[i] + ( ( c1 * rand1 ) * ( self.localbest[i] - self.currentposition[i]) ) + ( ( c2 * rand2 ) * ( globalbestposition_vector[i] - self.currentposition[i]) )
         #   tmp.append( newVelComponent );
         #   #print "{0} is the particles's {1}th velocity element".format( tmp, i );


         # TODO: verify via testing
         # In their BPSO algorithm, the velocity of each
         # individual particle Vi is a random number in the range of [0, 1]
         # generated at each iteration from which the position vector Xi
         # see: https://drive.google.com/file/d/0Bzwbc8cqElSAWEx3LVFJZmc4Ymc/view?usp=sharing
         self.velocity = [ round( random.random(), 3 ) for e in range( len(self.velocity) ) ];
       except:
          print "error in Particle.updatevelocity( ... )";

def particle_decoder(obj):
    '''Decode Particle types from arbitrary objects'''
    try:
        if '__type__' in obj and obj['__type__'] == 'Particle':
            return Particle( obj['currentposition'], obj['localbest'], obj['velocity'] )
        return obj
    except:
        print "error in particle_decoder( ... )"

def updateSwarmPopulation(elite1, swarmPopulation, fitness = [], Y_fitness = [], Yhat_fitness= [], c1 = .75, c2 = .75, bpso_lambda = 0.01, bpso_alpha = 0.5, bpso_beta = 0.05):
    try:
        '''TODO Update each Particle in the the binary particle swarm'''
        updatedSwarm   = []
        swarmSize     = len(swarmPopulation)
        local_fitness = np.array([e for e in fitness])
        for i in range( len(swarmPopulation) ):
            # decode serialized swarm particle
            jsonObj          = json.loads(swarmPopulation[i])
            particleInstance = json.loads('{"__type__": "Particle", "currentposition": ' + str(jsonObj['currentposition']) + ', "localbest": ' + str(jsonObj['localbest']) + ', "velocity": ' + str(jsonObj['velocity']) + '}', object_hook=particle_decoder)

            swarmParticle = particleInstance
            # 1) Update particle's velocity
            swarmParticle.updatevelocity( c1 = .75, c2 = .75 )
            # 2) Update particle's current position
            swarmParticle.updatecurrentposition( bpso_lambda = 0.01, bpso_alpha = 0.5, bpso_beta = 0.05)
            # 3) Update particle's local best position
            current_position_vector = [e for e in swarmParticle.currentposition];
            new_localbestposition   = rouletteWheelDescriptorSelection( current_position_vector, a = 0.75, k = 10 );
            swarmParticle.updatelocalbestposition([e for e in new_localbestposition]);

            updatedSwarm.append( swarmParticle.to_JSON() );

        return updatedSwarm
    except:
        print "error in updateSwarmPopulation( ... )"

def checkTerminationCriterion(currentIteration, maxIterationLimit = 30):
    '''Termination criterion is met (e.g. number of iterations performed, or adequate fitness reached)'''
    try:
         if (currentIteration >= maxIterationLimit):
            print "{0}No need to continue since fitness has not changed in the last {1} iterations".format(getLogTime(), currentIteration)
            return True;
         else:
            return False;
    except:
        print "error in evaluateHaltCondition"

def migrateParticleSwarmForNIterations(model, fileW, fitness, thisSwarmPopulation, elite1, elite1Index, TrainX, TrainY, ValidateX, ValidateY, TestX, TestY, unfit = 1000, numOfIterations = 200, Y_fitness = [], Yhat_fitness = [], c1 = .75, c2 = .75, swarmIterationLimit = 30, sessionid = 'fake_sessionid_'):
    '''Run the binary particle swarm optimization algorithm for the specified number of generations'''
    try:
        global globalbestfitness
        numOfPop                 = len(thisSwarmPopulation)
        numOfFea                 = len(globalbestposition_vector)
        evolutionHalt            = 0
        local_fitness            = np.array([e for e in fitness])
        local_Y_fitness          = np.array([e for e in Y_fitness]);
        local_Yhat_fitness       = np.array([e for e in Yhat_fitness]);

        # DUMMY value, I didn't put this here, take it out when possible, this shouldn't be needed
        local_c                  = 2
        for i in range(1, numOfIterations):
            #TODO: evaluate termination check, make sure we quit if the local fitness has not changed
            evolutionHalt = checkTerminationCriterion(i, maxIterationLimit = swarmIterationLimit)
            if evolutionHalt is True:
                break;

            publishLogMessage(queueId = sessionid, msg = "{0}This is swarm iteration {1}, minimum training fitness is: {2}".format(getLogTime(), i, round( globalbestfitness, 4)) );
            #print "{0}This is swarm iteration {1}, minimum training fitness is: {2}".format(getLogTime(), i, round( globalbestfitness, 4));
            if (local_fitness.min() < 0.005):
                print "{0}{1}\nGood: Fitness is low enough to quit\n{2}".format(getLogTime(), '*' * 35,'*' * 35);
                break;
            fittingStatus       = unfit
            while (fittingStatus == unfit):
                # find a new population
                updatedSwarmPopulation = updateSwarmPopulation(elite1, thisSwarmPopulation, fitness = local_fitness, Y_fitness = local_Y_fitness, Yhat_fitness = local_Yhat_fitness, c1 = .75, c2 = .75)
                fittingStatus, local_fitness, local_Y_fitness, local_Yhat_fitness, local_c = train_ml_model(model
                                                        , fileW
                                                        , updatedSwarmPopulation
                                                        , TrainX
                                                        , TrainY
                                                        , ValidateX
                                                        , ValidateY
                                                        , TestX
                                                        , TestY
                                                        , unfit = 1000
                                                        , generationLabel = i)
            # find the first elite position of the swarm population
            localSwarmElite, elite1Index     = findLocalBest(local_fitness, thisSwarmPopulation);
            updateGlobalBest(localSwarmElite = localSwarmElite, elite1Index = elite1Index, localEliteFitness = local_fitness[elite1Index]);
        
        return local_fitness.min();
    except:
        print "error in MoveForNIterations"
        return -1

def findFirstElitePosition(fitness, swarm_position_vectors):
    try:
        numOfPop    = swarm_position_vectors.shape[0]
        numOfFea    = swarm_position_vectors.shape[1]
        elite1      = zeros(numOfFea)
        elite1Index = 0
  
        for i in range(1, numOfPop):
           if (fitness[i] < fitness[elite1Index]):
               elite1Index = i

        for j in range(numOfFea):
            elite1[j] = swarm_position_vectors[elite1Index][j]

        return elite1, elite1Index
    except:
        print "error in findFirstElite"

def updateGlobalBest(localSwarmElite = [], elite1Index = -1, localEliteFitness = 1000):
    try:
        global globalbestposition_vector, globalbestfitness
        a                 = [e for e in localSwarmElite];
        b                 = [e for e in globalbestposition_vector];
        local_eval_vector = np.greater( a, b );
        # evaluate the local swarm's best position against the global best in all N-dimensions
        # of the problem space
        localSwarmElitePosition_eval = local_eval_vector[ (local_eval_vector == True) ];
        globalbestposition_eval      = local_eval_vector[ (local_eval_vector == False) ];
        # check if the local swarm's position is better than the current global best position
        # across all N-dimensions of the problem space
        if len(localSwarmElitePosition_eval) > len(globalbestposition_eval):
            # update global best position if this swarm 
            # cluster's local best position is better.
            globalbestposition_vector = np.array([ e for e in localSwarmElite]);
            globalbestfitness         = localEliteFitness;
            print "{0}Current minimum fitness: {1} is less than previous global fitness: {2}".format( getLogTime(), round(localEliteFitness,4), round(globalbestfitness,4) )
            print "{0}Promoting current minimum fitness to global fitness for next generations.".format( getLogTime() )
        else:
            pass;

        return;
    except:
        print "error in updateGlobalBest(...)"

def findLocalBest(fitness, swarm):
    '''Find the local best position in the swarm'''
    try:
        # collect current positions from encoded swarm particles
        tmpList = [];
        for i in range(len(swarm)):
            jsonObj          = json.loads(swarm[i])
            particleInstance = json.loads('{"__type__": "Particle", "currentposition": ' + str(jsonObj['currentposition']) + ', "localbest": ' + str(jsonObj['localbest']) + ', "velocity": ' + str(jsonObj['velocity']) + '}', object_hook=particle_decoder)
            positionVector   = np.array( [e for e in particleInstance.currentposition] )
            tmpList.append(positionVector)

        particle_position_vectors = np.array( [e for e in tmpList] );
        elite1, elite1Index       = findFirstElitePosition(fitness, particle_position_vectors)
        return elite1, elite1Index
    except:
        print "error in findLocalBest"

def rouletteWheelDescriptorSelection( current_position = [], a = 0.75, k = 10 ):
   try:
      '''
         A roulette wheel selection is used to construct
         descriptor subsets for each particle i where each descriptor is 
         assigned a portion of the roulette wheel based on its fractional 
         probability 'P' as determined where 'a' controls the selection pressure; 'a' > 1 emphasizes the 
         selection of highly fit descriptors, whereas 'a' < 1 increases the 
         chance of selecting less fit descriptors. To obtain the descriptor 
         subset, the roulette wheel is spun 'k' times, where 'k' is the 
         number of desired descriptors in the model.
         see Eq. 6: https://drive.google.com/file/d/0Bzwbc8cqElSAWEx3LVFJZmc4Ymc/view?usp=sharing
      '''
      # Note: careful not to raise a negative number to a fractional power
      tmp            = np.array( [ float( x**a ) if ( x > 0 or a.is_integer() ) else x for x in current_position] );
      denominator    = float(np.sum(tmp));
      newLocalBest   = np.array( [ ( float(x) / float(denominator) ) if float(denominator) > 0 else x for x in tmp ] );
      return newLocalBest;
   except:
      print 'error in rouletteWheelDescriptorSelection( ... )'

def createInitialSwarmPopulation(swarmSize, numOfFea, bpso_lambda = 0.01 ):
    '''Initializes swarm population with random positions in the search-space'''
    try:
        # Generate a random binary drug descriptor mask
        # represneting the spatial position of each particle in the swarm
        # NOTE: We persist the swarm population as Particle class instances 
        # since numpy has no straightforward way of representing 
        # arbitrary in-memory class objects.
        swarmpopulation = [];
        for i in range(swarmSize):
            # To keep the total number of selected descriptors low 
            # for each particle, we set lambda = 0.01.
            # see: https://drive.google.com/file/d/0Bzwbc8cqElSAWEx3LVFJZmc4Ymc/view?usp=sharing
            initial_velocityvector                = np.array( [ np.round( random.random(), 3 ) for i in range(numOfFea) ] );
            initial_position_vector               = np.array( [ 1 if v < bpso_lambda else 0 for v in initial_velocityvector ] );
            initial_localbestposition_vector      = rouletteWheelDescriptorSelection( initial_position_vector, a = 0.75, k = 10 );
            #In the initial generation, the velocity
            #and position vectors of the binary particle swarm are initialized 
            #according to the following rules
            # see Eq.10 and Eq.11: https://drive.google.com/file/d/0Bzwbc8cqElSAWEx3LVFJZmc4Ymc/view?usp=sharing
            newParticle = Particle(  currentposition = [ 1 if v < bpso_lambda else 0 for v in initial_velocityvector ]
                                   , localbest = [e for e in initial_localbestposition_vector]
                                   , velocity = [ np.round( random.random(), 3 ) for i in range(numOfFea) ])
            # serialize particle to JSON for distributed computing support
            swarmpopulation.append( newParticle.to_JSON() );

        #queueLogger.delay( sessionid = "default", msg = json.dumps(swarmpopulation) )
        return swarmpopulation
    except:
        print "error in createInitialPopulation"

def createMlr():
    try:
        return linear_model.LinearRegression();
    except:
        print "error creating linear_model.LinearRegression()";

def createSvr():
    try:
        return svm.SVR();
    except:
        print "error creating svm.SVR()";

def createKnn():
    try:
        return neighbors.KNeighborsRegressor(n_neighbors = 5, weights = 'distance');
    except:
        print "error creating neighbors.KNeighborsRegressor()";

def createRfr():
    try:
        return ensemble.RandomForestRegressor();
    except:
        print "error creating ensemble.RandomForestRegressor()";

def runSharedTask( *args ):
    '''
        Task runner
    '''
    try:
        jsonResultCollection = [];
        #TODO: Refactor
        taskname =  str(args[0]).strip()
        print " task '{0}' started:".format(taskname)
        if taskname == "":
            print "no task to run";
            pass;
        # creates the result output file
        elif taskname == "createAnOutputFile":
            file_name = str(args[1])
            taskid    = createAnOutputFile.delay(file_name).task_id;
            while createAnOutputFile.AsyncResult(taskid).state != 'SUCCESS' or createAnOutputFile.AsyncResult(taskid).state != 'FAILURE':
                if createAnOutputFile.AsyncResult(taskid).state == 'SUCCESS':
                    with allow_join_result():
                        outputFilePathRef = createAnOutputFile.AsyncResult(taskid).get();
                        print "task created output file '{0}' successfully".format( outputFilePathRef );
                        jsonResultCollection.append(outputFilePathRef)
                    break;
                elif createAnOutputFile.AsyncResult(taskid).state == 'FAILURE':
                    print "error creating output file";
                    break;
                else:
                    continue;
        # adds all of data from source file to a numpy array
        elif taskname == "placeDataIntoArray":
            # parse data from the file into the container array
            file_name      = str(args[1])
            taskid         = placeDataIntoArray.delay(file_name).task_id
            while placeDataIntoArray.AsyncResult(taskid).state != 'SUCCESS' or placeDataIntoArray.AsyncResult(taskid).state != 'FAILURE':
                if placeDataIntoArray.AsyncResult(taskid).state == 'SUCCESS':
                    print "successfully loaded data from source file '{0}'".format(file_name)
                    with allow_join_result():
                        tmpStr              = placeDataIntoArray.AsyncResult(taskid).get();
                        jsonIntermediate    = json.loads(tmpStr);
                        jsonResultCollection.append(jsonIntermediate)
                    break;
                elif placeDataIntoArray.AsyncResult(taskid).state == 'FAILURE':
                    print "failed to load data from source file: '{0}'".format(file_name)
                    break;
                else:
                    continue;
        else:
            pass;

        output = json.dumps( jsonResultCollection )
        print "done with task: " + taskname
        return output;
    except:
        print "error running shared task."

#main program starts in here
@shared_task
def run(sessionid = 'fake_sessionid_', trainXFile = 'Train-Data.csv', trainYFile = 'Train-pIC50.csv', crossValXFile = 'Validation-Data.csv', crossValYFile = 'Validation-pIC50.csv', testXFile = 'Test-Data.csv', testYFile = 'Test-pIC50.csv', output_filename = '', numFeatures = 385, swarmSize = 25, totalGenerations = 2000, unfit = 1000, R2req_train = 0.6, R2req_validate = 0.5, R2req_test = 0.5, iterationLimit = 30, scalingFactor = 0.9, mlName = "MLR"):
    try:
        global globalbestposition_vector
        mlTypeLookup  = {'MLR':createMlr, 'KNN': createKnn, 'SVR':createSvr, 'RFR':createRfr}
        localFileName = output_filename
        # default output file if output file name is invalid
        # Note: only uncomment when debugging from the shell, otherwise leave commented out.
        #if os.path.isfile(output_filename) == False:
        #    localFileName = "{}.csv".format( time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) );          
        #    localFileName = sessionid + localFileName;
        #else:
        #    pass;
        print 'output file: {0}'.format(output_filename)
        res                     = runSharedTask( "createAnOutputFile", localFileName )
        outputFilePathRef       = json.loads(res)[0].strip('"')

        # load training data
        res = runSharedTask( "placeDataIntoArray", trainXFile  )
        jsonX = json.loads(res)
        TrainX = np.array([e for e in jsonX[0]])

        res = runSharedTask( "placeDataIntoArray", trainYFile  )
        jsonY = json.loads(res)
        TrainY = np.array([e for e in jsonY[0]])

        # load cross-validation data
        res = runSharedTask( "placeDataIntoArray", crossValXFile  )
        jsonX = json.loads(res)
        ValidateX = np.array([e for e in jsonX[0]])

        res = runSharedTask( "placeDataIntoArray", crossValYFile  )
        jsonY = json.loads(res)
        ValidateY = np.array([e for e in jsonY[0]])

        # load test data
        res = runSharedTask( "placeDataIntoArray", testXFile  )
        jsonX = json.loads(res)
        TestX = np.array([e for e in jsonX[0]])

        res = runSharedTask( "placeDataIntoArray", testYFile  )
        jsonY = json.loads(res)
        TestY = np.array([e for e in jsonY[0]])

        # rescale training, cross-validation and test sets
        TrainX, ValidateX, TestX = rescaleTheData(TrainX, ValidateX, TestX)

        ## Generate selected model for initial binary particle swarm optimization agent population
        model = None;
        keyId = str(mlName).strip().upper();
        if mlTypeLookup.has_key( keyId ):
            print "Using machine learning model '{0}'".format(str(mlName).strip().upper());
            model = mlTypeLookup[keyId]();
        else:
            print "No valid machine learning model selected, defaulting to 'MLR'";
            model = mlTypeLookup['MLR']();

        fittingStatus = unfit
        # initialize global best with dummy data
        globalbestposition_vector = np.array( [float(-1) for i in range(numFeatures)] );
        localSwarmPopulation                                      = createInitialSwarmPopulation(swarmSize, numFeatures)
        fittingStatus, fitness, Y_fitness, Yhat_fitness, c        = train_ml_model(  model
                                                , outputFilePathRef
                                                , localSwarmPopulation
                                                , TrainX
                                                , TrainY
                                                , ValidateX
                                                , ValidateY
                                                , TestX
                                                , TestY
                                                , unfit           = 1000
                                                , generationLabel = 0)
        
        localSwarmElite, elite1Index     = findLocalBest(fitness, localSwarmPopulation)
        updateGlobalBest( localSwarmElite = localSwarmElite, elite1Index = elite1Index, localEliteFitness = fitness[elite1Index] );

        publishLogMessage(queueId = sessionid, msg = "{0}Completed generating initial random swarm population to train ""{1}"" model for Binary Particle Swarm Optimization".format(getLogTime(), (keyId if keyId != None else "")) );
        publishLogMessage(queueId = sessionid, msg = "{0}Base fitness position to optimize: {1}".format( getLogTime(), round(fitness.min(),4) ) );
        publishLogMessage(queueId = sessionid, msg = "{0}Starting binary particle swarm optimization".format( getLogTime() ) );
        
        bestFitness = migrateParticleSwarmForNIterations(model, outputFilePathRef, fitness, localSwarmPopulation, localSwarmElite, elite1Index, TrainX, TrainY, ValidateX, ValidateY, TestX, TestY, unfit = 1000, numOfIterations = totalGenerations, Y_fitness = Y_fitness, Yhat_fitness = Yhat_fitness, c1 = .75, c2 = .75, swarmIterationLimit = iterationLimit, sessionid = sessionid)

        # done with first proof-of-concept ML model conversion to a Celery distributed task :)
        print "successful end of model training"
        # return json serialized output file reference and fitness, for this model the optimal fitness from
        # training is returned from the migrateParticleSwarmForNIterations(...) function
        results = { "outputfile": "", "bestfitness": -1 }
        results["outputfile"]  = outputFilePathRef;
        results["bestfitness"] = round(bestFitness,3);
        output                 = json.dumps( results )
        return output
    except:
        print "error in main"
        return -1;