mkdir -p datasets

function download_speech_music_dataset {
  DIRECTORY="datasets/GTZAN Speech_Music"
  # GTZAN Speech_Music dataset
  if [ ! -d "$DIRECTORY" ]; then
    curl -L -o gtzan-musicspeech-collection.zip https://www.kaggle.com/api/v1/datasets/download/lnicalo/gtzan-musicspeech-collection
    unzip gtzan-musicspeech-collection.zip -d "$DIRECTORY"
    rm gtzan-musicspeech-collection.zip
    mv datasets/GTZAN\ Speech_Music/speech_wav datasets/GTZAN\ Speech_Music/speech
    mv datasets/GTZAN\ Speech_Music/music_wav datasets/GTZAN\ Speech_Music/music
  else
    echo "$DIRECTORY already exists."
  fi
}


# GTZAN Genre Dataset
function download_genres_dataset {
  DIRECTORY="datasets/GTZAN Genre"
  # GTZAN Speech_Music dataset
  if [ ! -d "$DIRECTORY" ]; then
    curl -L -o gtzan-dataset-music-genre-classification.zip https://www.kaggle.com/api/v1/datasets/download/andradaolteanu/gtzan-dataset-music-genre-classification
    unzip gtzan-dataset-music-genre-classification.zip -d "$DIRECTORY"
    rm gtzan-dataset-music-genre-classification.zip
  else
    echo "$DIRECTORY already exists."
  fi
}

download_speech_music_dataset &
download_genres_dataset &
wait
