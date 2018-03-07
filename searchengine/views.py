# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.shortcuts import render, render_to_response

# Create your views here.
from django.http import Http404, HttpResponse, HttpResponseRedirect
from django.views import View

from wine_recommender import get_wines

class searchView(View):
    def post(self, request):
        # <view logic>
        print(request.POST)
        # result = search(request.POST['term'])
        # [{"title": "Title Text", "content":"Content text", "links":"asdfasdfas"}]
        result = get_wines(request.POST['term'])
        # [{"title":request.POST['term'], "content":request.POST['term'] + " is awesome"}, 
        #           {"title":"Title 2", "content":"testing 1"}, 
        #           {"title":"Title 3", "content":'testing 2'}, 
        #         ]
        return render(request,'search.html',{'result': result})
    
    def get(self, request):
        return render(request,'search.html')