FOLDER=${1%/}
DATASET_NAME="${FOLDER##*/}"
MAX_PROCESSES=4
mkdir -p "EncodecDatasets/$DATASET_NAME"

for f in "$FOLDER"/*; do
  SUBFOLDER="${f##*/}"
  mkdir -p "EncodecDatasets"/"$DATASET_NAME"/"$SUBFOLDER"
  for item in "$FOLDER"/"$SUBFOLDER"/*; do
    FILE="${item##*/}"
    echo $FILE
    if [ $(jobs -r | wc -l) -ge 2 ]; then
      wait $(jobs -r -p | head -1)
    fi
    encodec -f "$item" "EncodecDatasets/$DATASET_NAME/$SUBFOLDER/${FILE%.*}.ecdc" &
  done
done

wait
