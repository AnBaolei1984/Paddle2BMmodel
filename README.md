# 1. Paddle2BMmodel使用方法

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


# 2. 模型拆分（如需要）

  对于一些后处理部分或其它算子不支持的模型，可以考虑将其进行拆分，模型前半部分转换成bmodel在TPU上运行，后半部分通过Paddle inference（X86）或Paddle-Lite(arm)在cpu上执行。
  
  拆分模型使用tools/segment_paddle_model.py工具。
  
  在工具的40行seg_node_names设置需要截断的node name（tensor的名字 ），可以通过netron工具查看，如下图。
  ![image](https://user-images.githubusercontent.com/49897975/123189571-a1563680-d4d0-11eb-8c1f-e74245be0c91.png)

  设置后，运行工具，会生成first_model和second_model两个模型，完成了原模型的拆分。

  
