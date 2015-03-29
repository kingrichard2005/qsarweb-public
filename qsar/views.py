import json
import os, tempfile, zipfile
import time
import uuid
import hashlib
import sys, traceback
import mimetypes
from django.shortcuts import get_object_or_404, render
from django.http import HttpResponse
from django.http import JsonResponse
from django.http import Http404
from django.http import HttpResponseRedirect
from django.template import RequestContext, loader
from django.core.urlresolvers import reverse
from django.views import generic
from django.core import serializers
from django.core.servers.basehttp import FileWrapper
from django.conf import settings
from qsar.models import *
from demoapp.tasks import *
# note: avoid importing all (*), this triggers a namespace conflict that causes an Attribute error to be thrown
from demo_bpso.tasks import run
from celery.result import allow_join_result
from celery.task.control import revoke
from pika import *
from helper import splitRawInput

class IndexView(generic.ListView):
    '''This returns the main index view under ./qsar/templates/qsar/'''
    queryset            = [];
    context_object_name = 'session'
    model               = Session
    template_name       = 'qsar/index.html'

    def dispatch(self, request, sessionid, *args, **kwargs):
        # If session id is set
        #if sessionid != None:
        #    # TODO: lookup session id if it's specified and return 
        #    # associated Session object in the (Index) view's queryset
        #    querysetqueryset = Session.objects.filter(sessionhash__exact = sessionid);
        #else:
        #    pass;
        return super(IndexView, self).dispatch(request, sessionid, *args, **kwargs)

    def get_context_data(self, **kwargs):
        context = super(IndexView, self).get_context_data(**kwargs)
        # If session id is set
        if self.kwargs['sessionid'] != None:
            # TODO: lookup session id if it's specified and return 
            # associated Session object in the (Index) view's queryset
            # We've got SESSIONS!! :-)
            context['session'] = Session.objects.filter(sessionhash__exact = self.kwargs['sessionid'])[0];
        else:
            pass;
        return context

def getMessageFromQueue(sessionid, taskid, dataSetFileList = []):
    '''
        Returns a dictionary object containing the key='statusmsg' 
        and key='outputfile', key='outputfile'.value will default 
        to '', if no output file's been produced.
    '''
    try:
        connection                   = pika.BlockingConnection()
        channel                      = connection.channel()
        strSessionId                 = str(sessionid)
        statusmsg                    = '';
        channel.queue_declare( queue = strSessionId )
        output                       = {}
        # Return not status if task hasn't started
        if run.AsyncResult(taskid).state == 'PENDING':
            output['statusmsg']  = 'no status to report';
            output['outputfile'] = ''
        # Get status of running task
        elif run.AsyncResult(taskid).state == 'STARTED':
            # If there are no new messages in the queue, then this
            # tuple will (None, None, None)
            method_frame, header_frame, body = channel.basic_get( queue = strSessionId )
            # hint 1: no new messages
            if method_frame == None:
                output['statusmsg']  = 'no status to report';
                output['outputfile'] = '';
            else:
                # hint 2: no new messages
                if method_frame.NAME != 'Basic.GetEmpty':
                    # (ack)nowledge receipt of the message
                    channel.basic_ack(delivery_tag=method_frame.delivery_tag)
                    statusmsg = body;
                    output['statusmsg']  = statusmsg
                    output['outputfile'] = ''
                else:
                    output['statusmsg']  = 'no status to report'
                    output['outputfile'] = ''
        # Remove file artifacts created from this training session, update the sessions output reference 
        # to refer to this file as the latest results.  Finally return the latest results
        # NOTE: for now the input file referenced in this session is preserved
        elif run.AsyncResult(taskid).state == 'SUCCESS':
            # Append return value to latest status
            with allow_join_result():
                taskResult           = run.AsyncResult(taskid).get();
                jsonResult           = json.loads(taskResult);
                # todo: save (valid) output file reference to QSAR database Output table,
                # then associate the reference with this session so users can download
                statusmsg            =  'Training completed successfully, best fitness: "{0}", result output file reference: {1}'.format(jsonResult['bestfitness'], str(jsonResult['outputfile']).replace('\\\\', '\\') );
                output['statusmsg']  = statusmsg
                output['outputfile'] = jsonResult['outputfile']
                # update session
                updateSession        = Session.objects.filter(sessionhash__exact = sessionid)[0];                    
                removeFilesFromTargetList(dataSetFileList);
                # save new input file entity to database
                newOutputFile        = Output(Output_File_Loc = jsonResult['outputfile']);
                newOutputFile.save();
                updateSession.Output = newOutputFile;
                updateSession.save();

        # cleanup and return status message
        connection.close();        
        return output
    except:
        print 'error in getMessageFromQueue( ... )'

def getLatestStatus(request, sessionid):
    try:
        taskid         = '-1' if request.POST["taskid"] == '' else request.POST["taskid"];
        msg            = 'no status to report'
        outputfilename = '';
        if sessionid != '':
            # only train under the context of a valid session
            if Session.objects.filter(sessionhash__exact = sessionid).exists():
                # get status message from model being trained
                dataSetFileList = json.loads(request.POST["dataSetFileList"]);
                output          = getMessageFromQueue(sessionid, taskid, dataSetFileList = dataSetFileList)
                if output['outputfile'] != '':
                    outputfilename = output['outputfile'];
                    outputfilename = outputfilename.replace('\\\\', '\\')
                    outputfilename = outputfilename.replace('\\\\', '\\')
                    msg    = "Latest training status: {0}".format(output['statusmsg']);
                else:
                    msg    = "Latest training status: {0}".format(output['statusmsg']);
            else:
                pass;

        return JsonResponse({'resultCode': str(0), 'data': msg, 'error': '', 'taskid': taskid, 'outputfile': outputfilename});
    except:
        return JsonResponse({'resultCode': str(-1), 'data': '','error': 'error while startng model training, please report to administrators.'});

def startTraining(request, sessionid):
    try:
        taskid          = '-1' if request.POST["taskid"] == '' else request.POST["taskid"];
        msg             = ''
        dataSetFileList = []
        
        # only train under the context of a valid session
        if sessionid != '':
            if Session.objects.filter(sessionhash__exact = sessionid).exists():
                # Handle training task execution
                # NOTE: Docs mention modifying default process limit to handle potentially
                # many results being queued.
                # see: http://celery.readthedocs.org/en/latest/userguide/tasks.html#states
                # see: http://celery.readthedocs.org/en/latest/userguide/tasks.html#task-result-backends
                # Start new task if one hasn't already been started
                thisSession       = Session.objects.filter(sessionhash__exact = sessionid)[0];
                inputFilePath     = thisSession.Input.Input_File_Loc if thisSession.Input != None else "";
                inputFullFilePath = os.path.join( os.getcwd(), inputFilePath );
                if os.path.isfile(inputFullFilePath):
                    if run.AsyncResult(taskid).state == 'PENDING':
                        dataSetFileList    = splitRawInput(inputFullFilePath, sessionid  = sessionid[:64]);
                        tmp                = str(sessionid)
                        filename           = r"{0}_{1}.csv".format( sessionid[:64], time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) );
                        outputFullFilePath = os.path.join( os.getcwd(), "qsar", "static", "qsar", "results", filename );
                        # DEBUG using default data set files in project root
                        #taskid             = run.delay(sessionid = tmp, output_filename = outputFullFilePath).task_id
                        taskid = run.delay(sessionid        = tmp
                                            ,trainXFile      = dataSetFileList[0]
                                            ,trainYFile      = dataSetFileList[1]
                                            ,crossValXFile   = dataSetFileList[2]
                                            ,crossValYFile   = dataSetFileList[3]
                                            ,testXFile       = dataSetFileList[4]
                                            ,testYFile       = dataSetFileList[5]
                                            ,output_filename = outputFullFilePath
                                            ).task_id
                        
                        msg    = 'Model training started under task id: "{0}".  Check for latest status using "Get Training Status"'.format(taskid)
                        # add input file reference
                        dataSetFileList.append(inputFullFilePath);
                            
                    elif run.AsyncResult(taskid).state == 'STARTED':
                        tmp    = getMessageFromQueue(sessionid, taskid)
                        msg    = 'Model training started under task id: "{0}".  Latest status is {1}, additional training status can be fetched using "Get Training Status"'.format(taskid, tmp)
                    # Start new task if last task completed successfully
                    elif run.AsyncResult(taskid).state == 'SUCCESS':

                        # For now, cleanup/remove all files from previous session 
                        # TODO: except the result/output file, the user should have a 
                        # reference to this file until this new task completes successfully.
                        dataSetFileList = json.loads(request.POST["dataSetFileList"]);
                        removeFilesFromTargetList(dataSetFileList);

                        dataSetFileList    = splitRawInput(inputFullFilePath, sessionid  = sessionid[:64]);
                        tmp                = str(sessionid)
                        filename           = r"{0}_{1}.csv".format( sessionid[:64], time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) );
                        outputFullFilePath = os.path.join( os.getcwd(), "qsar", "static", "qsar", "results", filename );
                        # DEBUG using default data set files in project root
                        #newtaskid          = run.delay(sessionid = tmp, output_filename = outputFullFilePath).task_id;
                        newtaskid = run.delay(sessionid        = tmp
                                               ,trainXFile      = dataSetFileList[0]
                                               ,trainYFile      = dataSetFileList[1]
                                               ,crossValXFile   = dataSetFileList[2]
                                               ,crossValYFile   = dataSetFileList[3]
                                               ,testXFile       = dataSetFileList[4]
                                               ,testYFile       = dataSetFileList[5]
                                               ,output_filename = outputFullFilePath
                                               ).task_id

                        msg                = 'Training completed successfully in a previous run as task id: "{0}".  New training task started with id: {1}.  Check latest status using "Get Training Status'.format(taskid, newtaskid)
                        taskid             = newtaskid
                        dataSetFileList.append(inputFullFilePath);
                    else:
                        #TODO: what should be reported to the user when a task fails?  Maybe just a simple message "Task failed: {last task message}, please restart your task"
                        pass;
                else:
                    msg = 'Please provide valid input data set before starting model training';
            else:
                msg = 'The specified session id {0}, could not be identified, please create a new session.'.format(sessionid);
        return JsonResponse({'resultCode': str(0), 'data': msg,'error': '', 'taskid': taskid, 'dataSetFileList': dataSetFileList});
    except:
        return JsonResponse({'resultCode': str(-1), 'data': '','error': 'error while startng model training, please report to administrators.'});

def stopTraining(request, sessionid):
    try:
        taskid         = '-1' if request.POST["taskid"] == '' else request.POST["taskid"];
        msg            = 'no status to report'
        outputfilename = '';
        if sessionid != '':
            # don't try to stop training if this session is not valid
            if Session.objects.filter(sessionhash__exact = sessionid).exists():
                dataSetFileList = [e for e in json.loads(request.POST["dataSetFileList"])];
                # revoke task
                revoke(taskid, terminate = True)
                # cleanup training artifacts from the halted session
                removeFilesFromTargetList(dataSetFileList)
                # Dissassiciate the deleted input file reference from the session entity
                # and remove the input file entity from the database
                updateSession       = Session.objects.filter(sessionhash__exact = sessionid)[0];
                instance            = updateSession.Input;
                updateSession.Input = None;
                updateSession.save();
                instance.delete();
            else:
                pass;

        return JsonResponse({'resultCode': str(0), 'data': msg, 'error': '', 'taskid': taskid, 'outputfile': outputfilename});
    except:
        return JsonResponse({'resultCode': str(-1), 'data': '','error': 'error while stoppng model training, please report to administrators.'});

def removeFilesFromTargetList(targetReferenceList):
    '''Helper removes (valid) files from the target list'''
    try:
        for i in range( 0, len(targetReferenceList) ):
            filepath = targetReferenceList[i];
            if os.path.isfile(filepath):
                os.remove(filepath);
    except:
        print 'error in removeDataSetFiles( ... )'

def saveInputFile(f, unique_storage_name):
    '''Saves raw file uploaded by user'''
    try:
        directory_location = os.path.join( os.getcwd(), 'qsar', 'static', 'qsar', 'uploads');
        directory_location = '\\'.join([directory_location, unique_storage_name]);
        with open(directory_location, 'wb+') as destination:
            for chunk in f.chunks():
                destination.write(chunk)
        return directory_location;
    except:
        print 'error in handle_uploaded_file()'

def uploadTrainingDataInput(request, sessionid):
    try:
        filename = '';
        if request.method == 'POST':
            files  = request.FILES
            # File upload handling...
            if sessionid != "":
                # use first 64-characters as part of the uploaded file nomenclature
                # this is better to identify uploaded files associated with a specific
                # session without including the entire 256 hash char session string in the file name
                filename            = '_'.join( [sessionid[:64], request.FILES['files[]'].name]);
                filePath            = saveInputFile(request.FILES['files[]'], filename)
                # save new input file entity to database
                newInputFile        = Input(Input_File_Loc = filePath);
                newInputFile.save();
                # update session
                updateSession       = Session.objects.filter(sessionhash__exact = sessionid)[0];
                updateSession.Input = newInputFile;
                updateSession.save();

        return JsonResponse({'resultCode': str(0), 'data': 'All good in the hood, your training data file has been uploaded! :)','error': ''});
    except:
        return JsonResponse({'resultCode': str(-1), 'data': '','error': 'error while upload training data input, please report to administrators.'});

def getResultFile(request, filename):
    try:
        PROJECT_PATH                     = os.path.abspath(os.path.dirname(__file__))
        local_filename                   = os.path.join(os.path.abspath(os.path.dirname(__file__)), "static", "qsar", "results", filename)
        # extract raw (r) string
        local_filename                   = r"".join(local_filename)
        download_name                    = r"".join(filename);
        wrapper                          = FileWrapper(open(local_filename));
        content_type                     = mimetypes.guess_type(local_filename)[0];
        response                         = HttpResponse(wrapper,content_type=content_type)
        response['Content-Length']       = os.path.getsize(local_filename)    
        response['Content-Disposition']  = "attachment; filename=%s"%download_name
        return response;
    except:
        print 'error in getResultFile';
      
def getModelTypes(request, sessionid):
    '''Gets distinct model type names from table.ml_model'''
    try:
        mlTypeList      = [model.Type_Name for model in Model.objects.distinct('Type_Name')];
        responseMessage = {'resultCode': str(0), 'sessionid': '', 'data': mlTypeList,'error': ''};
        if sessionid != None:
            # restore this sessions type name
            thisSession                    = Session.objects.filter(sessionhash__exact = sessionid)[0];
            responseMessage['Type_Name']   = thisSession.Model.Type_Name;
            responseMessage['Model_Name']  = thisSession.Model.Model_Name;
            responseMessage['sessionid']   = thisSession.sessionhash;
        else:
            pass;
        return JsonResponse(responseMessage);
    except:
        errorMessage = {'resultCode': str(-1), 'data': '','error': 'error while getting Model Types from database, please report to administrators.'}
        return JsonResponse(errorMessage);

def getMachineLearningModels(request, sessionid):
    '''Get machine learning models for selected type'''
    try:
        modelType                    = str(request.POST.get("type", "")).replace('"','');
        machineLearningModelNameList = [ model.Model_Name for model in Model.objects.filter( Type_Name = modelType ) ];
        responseMessage              = {'resultCode': str(0), 'data': machineLearningModelNameList,'error': ''};
        if sessionid != None:
            # restore this sessions type name
            thisSession                    = Session.objects.filter(sessionhash__exact = sessionid)[0];
            responseMessage['Model_Name']  = thisSession.Model.Model_Name;
            responseMessage['sessionid']   = thisSession.sessionhash;
        else:
            pass;
        return JsonResponse(responseMessage);
    except:
        return JsonResponse({'resultCode': str(-1), 'data': '','error': 'error while getting Machine Learning Models from database, please report to administrators.'});

def getEvolutionaryAlgorithms(request, sessionid):
    '''Get Evolutionary Algorithms for selected machine learning model'''
    try:
        #mlName                       = str(request.POST.get("mlName", "")).replace('"','');
        # note: there should only one model unique model name in the database, make sure this enforced
        # correctly with a unique key constraint on the ml_model_name database field
        #selectedMachineLearningModel = MlModel.objects.filter( ml_model_name = mlName )[:1];
        # get evolutionary algorithms used by this machine learning model
        #eaUseList = Uses.objects.filter( ml_model = selectedMachineLearningModel[0] );

        # Get evolutionary algorithms names
        # TODO: filter duplicates
        EvolutionaryAlgorithmsNameList = [ name_tuple[0] for name_tuple in Evo_Alg.objects.values_list('Evo_Alg_Name') ];
        responseMessage = {'resultCode': str(0), 'data': EvolutionaryAlgorithmsNameList,'error': ''}
        if sessionid != None:
            # restore this sessions type name
            thisSession                      = Session.objects.filter(sessionhash__exact = sessionid)[0];
            responseMessage['Evo_Alg_Name']  = thisSession.Evo_Alg.Evo_Alg_Name;
            responseMessage['sessionid']     = thisSession.sessionhash;
        else:
            pass;
        return JsonResponse(responseMessage);
    except:
        return 'error while getting Evolutionary Algorithms from database, please report to administrators.';

def getImplementationMethods(request, sessionid):
    '''Get Implementation Methods for evolutionary algorithms'''
    try:
        ImplementationMethodsList = [ im.Impl_Desc for im in Implementation.objects.distinct('Impl_Desc') ];
        responseMessage =  {'resultCode': str(0), 'data': ImplementationMethodsList,'error': ''};
        if sessionid != None:
            # restore this sessions type name
            thisSession                      = Session.objects.filter(sessionhash__exact = sessionid)[0];
            responseMessage['Impl_Desc']  = thisSession.Impl.Impl_Desc;
            responseMessage['sessionid']     = thisSession.sessionhash;
        else:
            pass;
        return JsonResponse(responseMessage);
    except:
        return 'error while getting Implementation Methods from database, please report to administrators.';

def saveSession(request, sessionid):
    '''Save the QSAR Session'''
    try:
        if request.method == 'POST':
            # get distinct model types from table.ml_model
            #mlTypeList = [model.Type_Name for model in Model.objects.distinct('type_name')]
            # Django gotcha, was manually setting primary key b/c
            # the class model.id attribute defaulted to "IntegerField" type after running
            # inspectdb to reverse engineer model classes from the Postgresql database
            # tables created from Matineh's, excellent , SQL script :)
            # see https://docs.djangoproject.com/en/dev/howto/legacy-databases/

            # FIX & NOTES: To auto-increment the pk, the class model.id attribute
            # must be of type "AutoField", this made it happy now we can create
            # instances of our entity models like normal classes in Python, then save
            # them to the database dynamically using Django's ORM.  Suggest "cleaning up"
            # autogenerated entity classes in models.py (e.g. changing IntegerField to AutoField
            # where needed) to make sure they accurately represent the entities defined
            # in the database.  Also, next steps will be to start defining the
            # rest of the client-side views for the QSAR Web Modellor. While we work on
            # the QSAR client-side dashboard, we should also define unit tests for functions we define
            # in views.py which will basically be to test the Web API we're creating.
            # 
            # Below are some suggested readings to better understand the entity-relation mapper 
            # for Django's MV* architecture:
            # https://docs.djangoproject.com/en/1.7/intro/tutorial01/#creating-models
            # https://docs.djangoproject.com/en/1.7/topics/db/models/
            # https://docs.djangoproject.com/en/1.7/topics/db/queries/
            # https://docs.djangoproject.com/en/dev/topics/testing/

            # how to save a qsar session
            # TODO: Make sure request parameters are valid
            # Collect corresponding database objects for user's configuration selections 
            implementationObj       = Implementation.objects.get(Impl_Desc = str( request.POST["imName"] ).replace('"',''));
            eaObj                   = Evo_Alg.objects.get(Evo_Alg_Name = str( request.POST["eaName"] ).replace('"',''));
            mlObj                   = Model.objects.filter(Model_Name = str( request.POST["mlName"] ).replace('"',''))[0]

            # Build a session object
            # note: Since the "sessionsalt" is a required attribute of the Session entity (i.e. it cannot be null), 
            # we just use a fake value for now 
            newSession              = Session(sessionsalt = 1, Impl = implementationObj, Evo_Alg = eaObj, Model = mlObj);
            newSession.sessionhash  = hashlib.sha256( serializers.serialize('json', [ newSession, ]) ).hexdigest()
            # This saves or updates the session object, if the "sessionhash" is already in the database, it simply gets updated
            newSession.save()
            # return the session id token to the calling client
            uniqueId               = newSession.sessionhash;
            newSessionMessage      = 'Your sessions been saved, your unique session ID is "{0}".  Have Fun!! :)'.format(uniqueId);
            return JsonResponse({'data': {'msg': newSessionMessage,'sessiondid': uniqueId}})
    except:
        print "error saving QSAR session from database, please report to administrators."