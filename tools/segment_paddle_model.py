import paddle
import paddle.fluid as fluid
import sys
import os

def fetch_tmp_vars(block, fetch_targets, var_names_list = None):
  """
  """
  if len(var_names_list) == 0:
    print('if select front part, fetch_targets should set node name!!!')
    exit(0)

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

def fluid_inference(model_path, image_size, is_front):
    exe = fluid.Executor(fluid.CPUPlace())
    [inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(model_path, exe, os.path.join(model_path, 'model'), os.path.join(model_path, 'params'))
    print ("----") 
    global_block = inference_program.global_block()
    if is_front == 1:
      ## [] set node name
      fetch_targets = fetch_tmp_vars(global_block, fetch_targets, ['squeeze_0.tmp_0']) 
    print (fetch_targets)
    with fluid.program_guard(inference_program):
      if is_front == 1:
        fluid.io.save_inference_model('./save_model', feeded_var_names = feed_target_names, target_vars=fetch_targets, executor = exe)
      else:
        fluid.io.save_inference_model('./save_model', feeded_var_names = ['squeeze_0.tmp_0'], target_vars=fetch_targets, executor = exe)

if len(sys.argv) < 5:
    raise NameError('Usage: python ./infer_image.py path/to/model input_width input_height is_front')

model_path = sys.argv[1]
input_width = int(sys.argv[2])
input_height = int(sys.argv[3])
is_front = int(sys.argv[4])
image_size = (input_width, input_height)
paddle.enable_static()
fluid_inference(model_path, image_size, is_front)
    
