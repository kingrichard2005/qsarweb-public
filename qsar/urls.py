from django.conf.urls import patterns, url

from qsar import views

urlpatterns = patterns('',
    # ex: /qsar/
    url(r'^((?P<sessionid>[0-9a-f]{64})/)?$', views.IndexView.as_view(), name='index',),
    # ex: /qsar/saveSession/
    url(r'^((?P<sessionid>[0-9a-f]{64})/)?saveSession/$', views.saveSession),
    # ex: /qsar/getModelTypes/
    url(r'^((?P<sessionid>[0-9a-f]{64})/)?getModelTypes/$', views.getModelTypes),
    # ex: /qsar/getMachineLearningModels/
    url(r'^((?P<sessionid>[0-9a-f]{64})/)?getMachineLearningModels/$', views.getMachineLearningModels),
    # ex: /qsar/getEvolutionaryAlgorithms/
    url(r'^((?P<sessionid>[0-9a-f]{64})/)?getEvolutionaryAlgorithms/$', views.getEvolutionaryAlgorithms),
    # ex: /qsar/getImplementationMethods/
    url(r'^((?P<sessionid>[0-9a-f]{64})/)?getImplementationMethods/$', views.getImplementationMethods),
    # ex: /qsar/uploadTrainingDataInput/
    url(r'^((?P<sessionid>[0-9a-f]{64})/)?uploadTrainingDataInput/$', views.uploadTrainingDataInput),
    # ex: /qsar/{sessionid}/startTraining/
    url(r'^((?P<sessionid>[0-9a-f]{64})/)?startTraining/$', views.startTraining),
    # ex: /qsar/{sessionid}/getLatestStatus/
    url(r'^((?P<sessionid>[0-9a-f]{64})/)?getLatestStatus/$', views.getLatestStatus),
    # ex: /qsar/{sessionid}/getLatestStatus/
    url(r'^((?P<sessionid>[0-9a-f]{64})/)?getResultFile/$', views.getResultFile),
    # read about this directory here : /static/qsar/results/AboutThisDirectory.md
    url(r'^static/qsar/results/(?P<filename>\w{0,64}_\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}\.(md|csv))$', views.getResultFile),
    # ex: /qsar/{sessionid}/stopTraining/
    url(r'^((?P<sessionid>[0-9a-f]{64})/)?stopTraining/$', views.stopTraining),
)