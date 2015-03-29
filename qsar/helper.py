import csv
import time
import math
import sys
import os


def placeDataIntoArray(fileName):
    '''Read uploaded data into memory'''
    try:
        from numpy import *        # provides complex math and array functions
        with open(fileName, mode='rbU') as csvfile:
            datareader = csv.reader(csvfile, delimiter=',', quotechar=' ')
            dataArray  = array([row for row in datareader], dtype=float64, order='C')

        if (min(dataArray.shape) == 1): # flatten arrays of one row or column
            return dataArray.flatten(order='C')
        else:
            return dataArray
    except:
        print 'error in placeDataIntoArray(...)'

def splitRawInput(sourceFileName, sessionid = "fakesession", cvSplit = .15, testSplit = .15):
    '''
        Creates separate training, cross-validation and test set
        files from raw input set uploaded by the user.  This
        uses the 70/15/15 rule, i.e. 70% of raw data is used for
        training, while 15% are used for both cross-validation and 
        test sets. Note, this function assumes a well-formatted input file,
        TODO: document valid input file format

        Returns a list of the locations of these three files on disk 
        relative to the qsar root application folder. The list of elements
        returned is in the following order [trainX, trainY, testX, testY, cvX, cvY]
    '''
    try:
        from numpy import *        # provides complex math and array functions
        # read raw data set
        dataArray          = placeDataIntoArray(sourceFileName);
        static_location    = os.path.join( os.getcwd(), 'qsar', 'static', 'qsar', 'uploads');
        rawData            = array( [e for e in dataArray], dtype=float64, order='C' );
        
        # The last column of the data set is assumed to be the target (y) pIC50 values,
        # we separate this data from the rest of the observation sets using Numpy's array
        # slicing syntax
        cvData          = rawData[ 0 : int(len(rawData) * cvSplit), : ]
        cv_pIC50        = cvData[ :, -1 ]
        cvData          = cvData[ :, :-1 ]
        # update raw data set, i.e. filter cv elements that were extracted
        rawData         = rawData[ int(len(rawData) * cvSplit) : , : ]
        
        testData        = rawData[ 0 : int(len(rawData) * testSplit), : ]
        test_pIC50      = testData[ :, -1 ]
        testData        = testData[ :, :-1 ]
        # use remaining elements for training data set
        trainData       = rawData[ int(len(rawData) * testSplit) : , : ]
        train_pIC50     = trainData[ :, -1 ]
        trainData       = trainData[ :, :-1 ]

        # write all files, any existing file of the same name is overwritten
        trainX = '\\'.join([static_location, '{0}_{1}_train.csv'.format(sessionid,time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())) ]);
        with file(trainX, 'wb+') as fileOut:
            fileW     = csv.writer(fileOut)
            for i in range( 0, trainData.shape[0] ):
                row = trainData[i]
                fileW.writerow(row);

        trainY = '\\'.join([static_location, '{0}_{1}_train_pIC50.csv'.format(sessionid,time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())) ]);
        with file(trainY, 'wb+') as fileOut:
            fileW     = csv.writer(fileOut)
            for i in range( 0, train_pIC50.shape[0] ):
                row = train_pIC50[i]
                fileW.writerow([row]);

        testX = '\\'.join([static_location, '{0}_{1}_test.csv'.format(sessionid,time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())) ]);
        with file(testX, 'wb+') as fileOut:
            fileW     = csv.writer(fileOut)
            for i in range( 0, testData.shape[0] ):
                row = testData[i]
                fileW.writerow(row);

        testY = '\\'.join([static_location, '{0}_{1}_test_pIC50.csv'.format(sessionid,time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())) ]);
        with file(testY, 'wb+') as fileOut:
            fileW     = csv.writer(fileOut)
            for i in range( 0, test_pIC50.shape[0] ):
                row = test_pIC50[i]
                fileW.writerow([row]);

        cvX = '\\'.join([static_location, '{0}_{1}_cv.csv'.format(sessionid,time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())) ]);
        with file(cvX, 'wb+') as fileOut:
            fileW     = csv.writer(fileOut)
            for i in range( 0, cvData.shape[0] ):
                row = cvData[i]
                fileW.writerow(row);

        cvY = '\\'.join([static_location, '{0}_{1}_cv_pIC50.csv'.format(sessionid,time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())) ]);
        with file(cvY, 'wb+') as fileOut:
            fileW     = csv.writer(fileOut)
            for i in range( 0, cv_pIC50.shape[0] ):
                row = cv_pIC50[i]
                fileW.writerow([row]);

        return [trainX, trainY, testX, testY, cvX, cvY]
    except:
        print 'error in splitRawInput( ... )'
        return []