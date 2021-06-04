mkdir ./build
cd ./build
cmake .. -DPADDLE_TO_UMODEL=ON
make
cd ..
rm -rf ./build
