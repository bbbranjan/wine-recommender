$(document).ready(function(){
    var slider = document.getElementById('test-slider');
    noUiSlider.create(slider, {
     start: [0, 1200],
     connect: true,
     step: 1,
     tooltips: true,
     orientation: 'horizontal', // 'horizontal' or 'vertical'
     range: {       
        'min': 0,
        'max': 1200
     },
    });
    $('#save-filters').click(function(event){
        console.log("Clicked");
        var selected = [];
        $('input[name="location[]"]:checked').each(function() {
            selected.push($(this).attr('value'));
        });
        var location = selected;
        var rating  =  $('#rating-range').val();
        var price_range = slider.noUiSlider.get();
        var sentiment = $('#select-sentiment').val();
        console.log(location,rating,price_range,sentiment);
        // $('#filter-form').submit();
        
        //TODO: Change endpoint in urls.py 
        //TODO: Create POST request either to existing endpoint or to someother endpoint.
        console.log({location:location,rating:rating,price_range:price_range})
        $.post('http://localhost:8000/searchengine/',{csrfmiddlewaretoken:'khzPqIZrB0L5yF2qaUyv493RkTigV20WAN3bHlf62EPYbA7tkeASseg8DXAsuudy',location:location,rating:rating,price_range:price_range,sentiment:sentiment},function(success){
        });
        $('#exampleModal').modal('toggle');
    });

});