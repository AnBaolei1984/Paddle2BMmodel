import paddle
import paddle.fluid as fluid
import numpy as np
import sys
import os
from PIL import Image
GLB_arg_name = ''

def fetch_tmp_vars(block, fetch_targets, var_names_list=None):
    """
    """
    print ("[[[[")
    print (var_names_list)
    def var_names_of_fetch(fetch_targets):
        var_names_list = []
        for var in fetch_targets:
            var_names_list.append(var.name)
        return var_names_list

    fetch_var = block.var('fetch')
    old_fetch_names = var_names_of_fetch(fetch_targets)
    new_fetch_vars = []
    for var_name in old_fetch_names:
        var = block.var(var_name)
        new_fetch_vars.append(var)
    i = len(new_fetch_vars)
    if var_names_list is None:
        var_names_list = block.vars.keys()
    for var_name in var_names_list:
        if var_name != '' and var_name not in old_fetch_names:
            var = block.var(var_name)
            new_fetch_vars.append(var)
            block.append_op(
                type='fetch',
                inputs={'X': [var_name]},
                outputs={'Out': [fetch_var]},
                attrs={'col': i})
            i = i + 1
    return new_fetch_vars

def fluid_inference(model_path, image_path, image_size):
    img = Image.open(image_path, 'r').convert('RGB')
    new_img = img.resize((image_size[0], image_size[1]), Image.BILINEAR)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img_mean = np.array(mean).reshape((3, 1, 1))
    img_mean = img_mean.astype(np.float32)
    img_std = np.array(std).reshape((3, 1, 1))
    img_std = img_std.astype(np.float32)

    np_img = np.array(new_img)
    np_img = np_img[np.newaxis, :]
    np_img = np.transpose(np_img, [0, 3, 1, 2])
    np_img = np_img.astype(np.float32)
    np_img *= img_std
    np_img -= img_mean
    np_img.tofile('input_ref_data.dat')

    #write data into txt file
    with open('img.txt', 'w') as f:
      for ele in np_img:
        for el in ele:
          for data_ in el:
            for data in data_:
              f.write(str(data))
              f.write(" ")

    exe = fluid.Executor(fluid.CPUPlace())
    [inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(model_path, exe, os.path.join(model_path, 'model'), os.path.join(model_path, 'params'))
    print ("----") 
    print (fetch_targets)
    global_block = inference_program.global_block()
    fetch_targets = fetch_tmp_vars(global_block, fetch_targets, [GLB_arg_name]) 
    print (fetch_targets)
    with fluid.program_guard(inference_program):
        results = exe.run(inference_program, feed={feed_target_names[0]: np_img}, fetch_list=fetch_targets, return_numpy=False)
        print (results)
        for result in results: 
            res = np.array(result).flatten()
            res.tofile('output_ref_data.dat')
            out_num = min(len(res), 10000000000)
            fo = open("out_fluid.txt", "w")
            for i in range(0, out_num, 1):
                val = '%.6f ' % res[i]
                fo.write(str(val))
                fo.write("\n")
            fo.close()
        fluid.io.save_inference_model('./save_model', feeded_var_names = ['squeeze_0.tmp_0'], target_vars=fetch_targets, executor = exe)

if len(sys.argv) < 5:
    raise NameError('Usage: python ./infer_image.py path/to/model path/to/image image_width image_height')

model_path = sys.argv[1]
image_path = sys.argv[2]
image_width = int(sys.argv[3])
image_height = int(sys.argv[4])
if len(sys.argv) == 6:
    GLB_arg_name = sys.argv[5]
image_size = (image_width, image_height)
paddle.enable_static()
fluid_inference(model_path, image_path, image_size)
    
