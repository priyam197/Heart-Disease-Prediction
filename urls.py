from django.contrib import admin
from django.urls import path

import mlapp.views

urlpatterns = [
    path("",mlapp.views.home),
    path("about/",mlapp.views.about),
    path("myui/",mlapp.views.myui),
    path("myapi/",mlapp.views.myapi),
    path("ourteam/",mlapp.views.ourteam),
    path("predict/",mlapp.views.predict),
    path("contact/",mlapp.views.contact),
    path("about/",mlapp.views.about),
    path("finalresult/",mlapp.views.finalresult),

]