mkdir samples

curl -L -o samples/audio1.ogg https://github.com/librosa/data/raw/refs/heads/main/audio/198-209-0000.hq.ogg &
curl -L -o samples/audio2.ogg https://github.com/librosa/data/raw/refs/heads/main/audio/147793__setuniman__sweet-waltz-0i-22mi.hq.ogg &
curl -L -o samples/audio3.ogg https://github.com/librosa/data/raw/refs/heads/main/audio/5703-47212-0000.hq.ogg &
curl -L -o samples/audio4.ogg https://github.com/librosa/data/raw/refs/heads/main/audio/Kevin_MacLeod_-_Vibe_Ace.hq.ogg &

wait

ffmpeg -i samples/audio1.ogg -i samples/audio2.ogg -i samples/audio3.ogg -i samples/audio4.ogg -filter_complex '[0:0][1:0][2:0][3:0]concat=n=4:v=0:a=1[out]' -map '[out]' samples/concatenated.mp3 -y -v quiet -stats
ffmpeg -i samples/audio1.ogg samples/audio1.mp3 -y -v quiet -stats && rm samples/audio1.ogg &
ffmpeg -i samples/audio2.ogg samples/audio2.mp3 -y -v quiet -stats && rm samples/audio2.ogg &
ffmpeg -i samples/audio3.ogg samples/audio3.mp3 -y -v quiet -stats && rm samples/audio3.ogg &
ffmpeg -i samples/audio4.ogg samples/audio4.mp3 -y -v quiet -stats && rm samples/audio4.ogg &
wait