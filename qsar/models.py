# This is an auto-generated Django model module.
# You'll have to do the following manually to clean this up:
#   * Rearrange models' order
#   * Make sure each model has one field with primary_key=True
#   * Remove `managed = False` lines if you wish to allow Django to create, modify, and delete the table
# Feel free to rename the models, but don't rename db_table values or field names.
#
# Also note: You'll have to insert the output of 'django-admin.py sqlcustom [app_label]'
# into your database.
from __future__ import unicode_literals
from django.db import models
import datetime

class Implementation(models.Model):
    Impl_ID   = models.AutoField(primary_key=True)
    Impl_Desc = models.CharField(max_length=200, blank=True)

    class Meta:
        managed = True
        db_table = 'Implementation'

class Evo_Alg(models.Model):
    Evo_Alg_ID   = models.AutoField(primary_key=True)
    Evo_Alg_Name = models.CharField(max_length=50, blank=True)
    Evo_Alg_Desc = models.CharField(max_length=200, blank=True)

    class Meta:
        managed = True
        db_table = 'Evo_Alg'

class Input(models.Model):
    Input_ID       = models.AutoField(primary_key=True)
    Input_File_Loc = models.CharField(max_length=200, blank=True)

    class Meta:
        managed = True
        db_table = 'Input'

class Model(models.Model):
    model_id     = models.AutoField(primary_key=True)
    Model_Name   = models.CharField(max_length=50, blank=True)
    Type_Name    = models.CharField(max_length=20, blank=True)
    
    class Meta:
        managed  = True
        db_table = 'Model'

class Output(models.Model):
    Output_ID       = models.AutoField(primary_key=True)
    Output_File_Loc = models.CharField(max_length=200, blank=True)

    class Meta:
        managed = True
        db_table = 'Output'

class Session(models.Model):
    Session_ID    = models.AutoField(primary_key=True)
    sessionhash   = models.CharField(max_length=512, blank=True)
    sessionsalt   = models.CharField(max_length=512, blank=True)
    Creation_Date = models.DateTimeField('Creation_Date', default=datetime.datetime.now)
    Input         = models.ForeignKey('Input', blank=True, null=True)
    Output        = models.ForeignKey('Output', blank=True, null=True)
    Impl          = models.ForeignKey('Implementation', blank=True, null=True)
    Evo_Alg       = models.ForeignKey('Evo_Alg', blank=True, null=True)
    Model         = models.ForeignKey('Model', blank=True, null=True)

    class Meta:
        managed = True
        db_table = 'Session'