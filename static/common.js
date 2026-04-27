var audiosCache = {};
var audiosPosition = {};
var currentAudio;
var stopTimer;
var playbackInterval;
var currentPlaybackLine;


function play(event, recording) {
    var rect = event.target.getBoundingClientRect();
    var x = event.clientX - rect.left; //x position within the element.
    var width = rect.width;
    var percentage = x / width;

    if (currentAudio != null) {
        currentAudio.pause();
        hidePlaybackLine();
    }
    if (stopTimer) {
        clearTimeout(stopTimer);
    }
    if (playbackInterval) {
        clearInterval(playbackInterval);
    }
    
    if (audiosCache[recording] == null) {
        audiosCache[recording] = new Audio("/static/training-data/" + recording);
    }

    currentAudio = audiosCache[recording];
    
    // Show playback line at clicked position
    var recordingCard = event.target.closest('.recording');
    currentPlaybackLine = recordingCard.querySelector('.playback-line');
    if (currentPlaybackLine) {
        currentPlaybackLine.style.left = (percentage * 100) + '%';
        currentPlaybackLine.classList.add('active');
    }
    
    currentAudio.play().then(function() {
        currentAudio.currentTime = currentAudio.duration * percentage;
        audiosPosition[recording] = currentAudio.currentTime;
        
        // Update playback line position while playing
        playbackInterval = setInterval(function() {
            if (currentAudio && !currentAudio.paused) {
                var progress = currentAudio.currentTime / currentAudio.duration;
                if (currentPlaybackLine) {
                    currentPlaybackLine.style.left = (progress * 100) + '%';
                }
            }
        }, 50);
        
        stopTimer = setTimeout(function() {
            audiosCache[recording].pause();
            clearInterval(playbackInterval);
            hidePlaybackLine();
        }, 2000);
    }).catch(function(){
        console.error("Audio error");
    });
}

function hidePlaybackLine() {
    if (currentPlaybackLine) {
        currentPlaybackLine.classList.remove('active');
    }
    currentPlaybackLine = null;
}
