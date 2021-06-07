# Paddle2BMmodel

将paddle模型转化成可以在BM1684芯片上运行的fp32bmodel，或者量化需要的fp32.umodel的工具

1. 下载inference lib

    链接: https://pan.baidu.com/s/1KIYZ1Uvr7cJH7QKwlwZSgg  密码: 8oat
    
    下载后拷贝到Paddle2BMmodel目录下
  
2. 在比特大陆bmnnsdk2-bm1684目录，执行下面的命令（主要是设置环境变量，每开一次终端需要重新执行）：

        cd scripts

        source envsetup_cmodel.sh
  
3.  编译 

    a) 编译转换bmodel工具p2b
    
        cd Paddle2BMmodel
    
        sh tools/build_p2b.sh 

      编译成功后生成p2b的可执行文件

    b) 编译转换umodel工具p2u
    
        cd Paddle2BMmodel
    
        sh tools/build_p2u.sh 

      编译成功后生成p2u的可执行文件
  
4. 在config.txt设定模型的输入shape和路径

   多个输入的情况，不同shape以:为分隔符, 如1,3,608,608:300,300

5. 执行 ./p2u config.txt 或./p2b config.txt就可以完成转换

   
  
