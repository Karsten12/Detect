# Script to delete all captured frames
DESIRED_DIR="output"
CURRENT_DIR=${PWD##*/} 

if [ "$DESIRED_DIR" == "$CURRENT_DIR" ]
then
    echo Deleting pics
    # find . -type f -name '*.png' -exec rm {} +
else
    echo Incorrect directory
fi