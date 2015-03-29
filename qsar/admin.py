from django.contrib import admin

# Register your models here.
from qsar.models import *

admin.site.register(Implementation)
admin.site.register(Evo_Alg)
admin.site.register(Input)
admin.site.register(Output)
admin.site.register(Model)
admin.site.register(Session)
