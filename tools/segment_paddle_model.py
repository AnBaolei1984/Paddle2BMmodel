import paddle
import paddle.fluid as fluid
import sys
import os

def fetch_tmp_vars(block, fetch_targets, var_names_list = None):
  """
  """
  def var_names_of_fetch(fetch_targets):
    var_names_list = []
    for var in fetch_targets:
      var_names_list.append(var.name)
    return var_names_list

  fetch_var = block.var('fetch')
  old_fetch_names = var_names_of_fetch(fetch_targets)
  new_fetch_vars = []

  i = 0
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

def fluid_inference(model_path):
    exe = fluid.Executor(fluid.CPUPlace())
    [inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(model_path, exe, os.path.join(model_path, 'model'), os.path.join(model_path, 'params'))
    global_block = inference_program.global_block()
    #seg_node_names = ['squeeze_0.tmp_0']
    seg_node_names = []

    if len(seg_node_names) == 0:
      print('Segment node names must be set!!!')
      exit(0)

    second_fetch_targets = fetch_tmp_vars(global_block, fetch_targets, seg_node_names)
    with fluid.program_guard(inference_program):
      fluid.io.save_inference_model('./first_model', feeded_var_names = feed_target_names, target_vars = second_fetch_targets, executor = exe)
      fluid.io.save_inference_model('./second_model', feeded_var_names = seg_node_names, target_vars = fetch_targets, executor = exe)
    print ("The model was segmented successful :) :) :)") 

if len(sys.argv) < 2:
    raise NameError('Usage: python ./infer_image.py path/to/model')

model_path = sys.argv[1]
paddle.enable_static()
fluid_inference(model_path)

