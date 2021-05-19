# Paddle2BMUmodel

将paddle模型转化成可以在BM1684芯片上做量化的umodel的工具

1. 下载inference lib

    链接: https://pan.baidu.com/s/1PLsGHHzozcyAcqATyUkHvA  密码: dtt5
    
    下载后拷贝到，当前工程目录下
  
2. 在比特大陆bmnnsdk2-bm1684目录，执行下面的命令（主要是设置环境变量，每开一次终端需要重新执行）：

    cd scripts

    source envsetup_cmodel.sh
  
3. 在Paddle2BMmodel工程目录下执行，编译成功后生成p2u的可执行文件

    sh build.sh 
  
4. 在config.txt设定模型的输入shape和路径

5. 执行 ./p2u config.txt 就可以完成转换

   
  
