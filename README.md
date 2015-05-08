### CalState San Marcos Computer Science Department QSAR Web Modeling System
=======
Description:

This is the public repository for the CSUSM QSAR Web Modelling graduate project 2014-2015.  The CSUSM QSAR Web Modelling System is designed to be a scalable web application framework used to train quantitative structure-activity relationship (QSAR) models for purposes of identifying unique optimal physicochemical properties in bio-pharmaceutical chemical drug compound data sets.

Requirements:
* Anaconda Python 2.7 by Continuum Analytics
* Django 1.7
* Pika Python-driver for RabbitMQ
* RabbitMQ 
* Celery 3.1 Distributed Task Queue
* PostgreSQL 9.3 or greater

Pre-Reqs:

1. Make sure to create a new empty database in PostgreSQL called "qsar" without quotes, this is the QSAR web application's main database repository.
2. Make sure to have the RabbitMQ setup and configured (see http://www.rabbitmq.com/)
3. Make sure to have the Celery Distributed Task Queue setup and configured (see GETTING STARTED on the Celery homepage http://www.celeryproject.org/install/).
4. Make sure to install the Pika Python-driver for RabbitMQ (see https://pika.readthedocs.org/en/0.9.14/).  Pika is used for logging by models, a training model publishes messages to a (uniquely named) queue, clients subcribe to these queues to retrieve the status results from their models training session.  
 * Note: Since Celery is an enhancement of a special use case of RabbitMQ (see https://www.rabbitmq.com/tutorials/tutorial-two-python.html), we use it primarily for encapsulating training model tasks, while the standalone Pika driver will be used essentially just for generic logging purposes.

QuickStart

1. From the project root, register the QSAR application for synchronization with the QSAR database: ```python manage.py makemigrations qsar```
2. Next synchronize the QSAR web project and containing applications by entering the following: ```python manage.py syncdb```.  This will create the core QSAR model tables along with additional support tables in the QSAR database used for the built-in administrative interface.
  * Note: you may get a RunTimeWarning regarding the  Session.Creation_Date column, this is safe to ignore for now.
  * Following the creation of the database tables, Django will also ask if you want to create a superuser account for the adminitrative web interface, this step is optional and recommended for developer access to the database from the web browser UI.
3. Make sure to seed/populate the QSAR database tables Implementation, Evo_Alg and Model with the configuration options for user's training models.  You can run the sample SQL script under /demoresources/example_configurations.sql to add sample values to these tables for demonstration purposes, alternatively you could add configurations using the client-side admin interface. 
4. From the project root, run the command ```celery -A qsarweb worker --loglevel=Info```, this starts the Celery task server and loads the list of supported model configurations as Celery shared_tasks to be executed concurrently by Celery task workers.
5. Last run ```python manage.py runserver``` to start the Django development server.
  *  You can enter *127.0.0.1:8000/qsar* to navigate to the main dashboard (in development)
  *  Or you can enter *127.0.0.1:8000/admin* to navigate to the main administrative login, note this will require a user account be created as described in step 2, point 2.

Debugging Celery Tasks

Note: Make sure to execute reference commands without quotes 
* With RabbitMQ installed first, follow instructions to install RabbitMQ management tools
    which provides access to the QSAR task worker message broker interface to inspect various
    the progress & results of completed worker tasks at an administrative level.
    (see: http://www.rabbitmq.com/management.html )
* "python manage.py shell" - is used from the project root to run / debug Celery tasks manually, i.e. instead of from the QSAR website (see Celery Documentation for more info )
 * when developing/debugging it's good practice to manually clear/purge your RabbitMQ message queues periodically to avoid accumulating unacked messages.
* To do this on Windows
 1. First stop the RabbitMQ Windows service from the Windows Service Manager.
 2. Next, delete the default RabbmitMQ database file, on Windows this is created by default under 'C:\Users\{Your_Username}\AppData\Roaming\RabbitMQ\', just delete the folder in this directory ending in "-mnesia".  
 3. Finally restart the RabbitMQ Windows service, if you have the management tools installed you can see from the interface all exchanges, queues and messages have been cleared.
