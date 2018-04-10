# Create your views here.
# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.shortcuts import render, render_to_response


from django.http import Http404, HttpResponse, HttpResponseRedirect , JsonResponse
from django.views import View

from wine_recommender import get_wines
from autocorrect import spell

import time

class searchView(View):

    # Handle POST requests
    def post(self, request):

        # Start timer
        start_time = time.time()

        query = ' '.join([spell(word) for word in request.POST['term'].split()])

        # Debug request POST
        print(request.POST)
        
        # Get recommended wines 
        result = get_wines(query)

        time_taken = time.time() - start_time

        print(time_taken)
        
        # Render frontend based on results obtained
        return render(request,'search.html',{'query': query, 'result': result,'locations':['Singapore','India','China','Poland']})
    
    # Handle GET requests
    def get(self, request):
        #TODO: Get dynamic list of locations
        return render(request,'search.html',{'locations':['Singapore','India','China','Poland']})

class filterView(View):

    # Handle POST requests
    def post(self, request):
        
        # Debug request POST
        print(request.POST)

        # Render frontend based on results obtained
        return JsonResponse({'result':True})

