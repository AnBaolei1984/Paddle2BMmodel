mkdir ./build
cd ./build
cmake .. -DPADDLE_TO_BMODEL=ON -DPADDLE_TO_UMODEL=OFF
make
cd ..
rm -rf ./build
