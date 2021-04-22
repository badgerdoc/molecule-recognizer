export FONTS_FOLDER="fonts"
export ARCHIVE_NAME="fonts_archive.zip"

mkdir $FONTS_FOLDER

cd $FONTS_FOLDER
gdown "https://drive.google.com/uc?export=download&confirm=zwbh&id=1o1ZM5p3ow2hedYCy7X85trcYj4_8mHO3" -O $ARCHIVE_NAME
unzip -j $ARCHIVE_NAME
rm $ARCHIVE_NAME
cd -
