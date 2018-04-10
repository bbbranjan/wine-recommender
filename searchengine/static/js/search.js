$(document).ready(function(){
    var slider = document.getElementById('test-slider');
    noUiSlider.create(slider, {
     start: [20, 80],
     connect: true,
     step: 1,
     tooltips: true,
     orientation: 'horizontal', // 'horizontal' or 'vertical'
     range: {
       'min': 0,
       'max': 100
     },
    });
    $('#save-filters').click(function(event){
        console.log("Clicked");
        var location = $('input[name=location]:checked').val(); 
        var rating  =  $('#rating-range').val();
        var price_range = slider.noUiSlider.get();
        console.log(location,rating,price_range);
        // $('#filter-form').submit();
        
        //TODO: Change endpoint in urls.py 
        //TODO: Create POST request either to existing endpoint or to someother endpoint.
        $.post('searchengine',{location:location,rating:rating,price_range:price_range},function(success){
        });
        $('#exampleModal').modal('toggle');
    });

});