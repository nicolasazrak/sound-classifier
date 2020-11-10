import os


def get_raw_recordings():
    recordings = []
    for file in os.listdir("training-data/raw/"):
        if file.endswith(".wav"):
            recordings.append(file)
    recordings.sort()
    return recordings


def generate_thumbs():
    for recording in get_raw_recordings():
        if not os.path.isfile(os.path.join("training-data", "raw", recording + ".png")):
            print("Generating training-data/raw/{recording}.png")
            os.system(
                f'ffmpeg -i "training-data/raw/{recording}"  -filter_complex "[0:a]aformat=channel_layouts=mono,  compand=gain=10,  showwavespic=s=600x120:colors=#9cf42f[fg];  color=s=600x120:color=#44582c,  drawgrid=width=iw/10:height=ih/5:color=#9cf42f@0.1[bg];  [bg][fg]overlay=format=auto,drawbox=x=(iw-w)/2:y=(ih-h)/2:w=iw:h=1:color=#9cf42f" -frames:v 1 "training-data/raw/{recording}.png"')
