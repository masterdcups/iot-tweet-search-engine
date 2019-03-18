/*document.addEventListener('DOMContentLoaded', function () {
    var elems = document.querySelectorAll('select');
    var instances = M.FormSelect.init(elems, {});
    var tabs = document.querySelectorAll('.tabs');
    var instance = M.Tabs.init(tabs, {});

    var images = document.querySelectorAll('.movie-poster');

    for (var i = 0; i < images.length; i++) {
        var vibrant = new Vibrant(images[i]);
        var swatches = vibrant.swatches();
        var rgb = swatches.DarkVibrant.rgb;
        images[i].style.boxShadow = 'rgba('+rgb[0]+', '+rgb[1]+', '+rgb[2]+', 0.54) 0px 3px 17px 4px';
    }
});*/


$('.fav_button').click(function () {
    var element = $(this);
    $.ajax({
        url: "/mark_view",
        method: 'post',
        data: {
            tweet_id: $(this).data('tweet-id'),
        },
        success: function (result) {
            console.log(result);
            console.log(element);
            if (result === 'added') {
                element.removeClass('far');
                element.addClass('fas');
            }
            if (result === 'removed') {
                element.removeClass('fas');
                element.addClass('far');
            }
        }
    });
});