from __future__ import absolute_import

from celery import shared_task, Task
from celery.utils.log import get_task_logger
from django.core.cache import cache
import pika
import time

@shared_task
def AddTask(x, y, sessionid):
    '''This is a stub task for testing debugging distributed task queues'''
    try:
        msg        = "This is a status message from your training model";
        # this opens a channel to the RabbitMQ
        connection = pika.BlockingConnection()
        channel    = connection.channel()
        # we declare a new unique queue, identifiable by the session id, in 
        # the open RabbitMQ channel in order to allow us to publish (log) messages
        # to be read by subscribing clients with access to the session id.
        tmp        = str(sessionid)
        channel.queue_declare( queue = tmp )
        channel.basic_publish(exchange='', routing_key = tmp, body=msg)
        connection.close()

        # task result
        return x + y
    except:
        print 'error in AddTask(Task))'