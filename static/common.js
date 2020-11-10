var audiosCache = {};
var audiosPosition = {};
var currentAudio;
var stopTimer;


function play(event, recording) {
    var rect = event.target.getBoundingClientRect();
    var x = event.clientX - rect.left; //x position within the element.
    var width = rect.width;
    var percentage = x / width;

    if (currentAudio != null) {
        currentAudio.pause();
    }
    if (stopTimer) {
        clearTimeout(stopTimer);
    }
    
    if (audiosCache[recording] == null) {
        audiosCache[recording] = new Audio("/static/training-data/" + recording);
    }

    currentAudio = audiosCache[recording];
    currentAudio.play().then(function() {
        currentAudio.currentTime = currentAudio.duration * percentage;
        audiosPosition[recording] = currentAudio.currentTime;
        stopTimer = setTimeout(() => audiosCache[recording].pause(), 2000);
    }).catch(function(){
        console.error("Audio error");
    });
}