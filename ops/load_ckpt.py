import tensorflow as tf
slim = tf.contrib.slim

def load_st(sess, args):
    exclusions = []
    if args.checkpoint_exclude:
        exclusions = [scope.strip() for scope in args.checkpoint_exclude.split(',')]

    restore_var = []
    for var in tf.trainable_variables():
        excluded = False
        for exclusion in exclusions:
            if exclusion in var.name:
                excluded = True
                break
        if not excluded:
            restore_var.append(var)

    tf.train.Saver(var_list=restore_var).restore(sess, args.restore_from)
    print("Restored model parameters from {}".format(args.restore_from))


def load_mt(sess, args):
    '''Load trained weights for multi-task networks.

    Args:
      sess: TensorFlow session.
      args: arguments.
    '''
    checkpoint_dir_list = []
    if args.restore_from_1 is not None and args.restore_from_2 is not None:
        checkpoint_dir_list.append(args.restore_from_1)
        checkpoint_dir_list.append(args.restore_from_2)
    else:
        return None

    exclusions = []
    if args.checkpoint_exclude:
        exclusions = [scope.strip() for scope in args.checkpoint_exclude.split(',')]

    checkpoint_path_list = []
    for checkpoint_path in checkpoint_dir_list:
        if tf.gfile.IsDirectory(checkpoint_path):
            checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)

        print('Fine-tuning from %s' % checkpoint_path)
        checkpoint_path_list.append(checkpoint_path)

    replace_from = args.replace_from
    replace_to_list = args.replace_to_list.split(',')

    if 'nddr' in args.network:
        _renamed_restore(checkpoint_path_list, replace_from, replace_to_list, exclusions, sess)
    else:
        _renamed_restore_no_fuse(checkpoint_path_list, replace_from, replace_to_list, exclusions, sess)


def _renamed_restore(checkpoint_dir_list, replace_from, replace_to_list, exclusions, sess):
    for checkpoint_dir, replace_to in zip(checkpoint_dir_list, replace_to_list):
        replaced_name_list = {}
        for var_name, _ in tf.contrib.framework.list_variables(checkpoint_dir):
            excluded = False
            for exclusion in exclusions:
                if exclusion in var_name:
                    excluded = True
                    break

            if excluded == True:
                continue

            new_name = var_name
            sp = new_name.split('/')
            if len(sp) > 1 and sp[1] != []:
                if 'fc' not in sp[1]:
                    num_layer = sp[1][-1]
                    old_num_layer = num_layer
                else:
                    num_layer = str(int(old_num_layer) + 1)
                new_name = new_name.replace(replace_from, replace_to + '_' + num_layer)
            else:
                continue

            new_var_to_restore = slim.get_variables_by_name(new_name)
            if len(new_var_to_restore) == 1:
                print('load ' + new_var_to_restore[0].name)
                replaced_name_list[var_name] = new_var_to_restore[0]

        restorer = tf.train.Saver(replaced_name_list)
        restorer.restore(sess, checkpoint_dir)


def _renamed_restore_no_fuse(checkpoint_dir_list, replace_from, replace_to_list, exclusions, sess):
    assert checkpoint_dir_list[0] == checkpoint_dir_list[1]
    checkpoint_dir = checkpoint_dir_list[0]
    # replaced_name_list = {}
    for var_name, _ in tf.contrib.framework.list_variables(checkpoint_dir):
        excluded = False
        for exclusion in exclusions:
            if exclusion in var_name:
                excluded = True
                break

        if excluded == True:
            continue

        new_names = []
        if 'fc' not in var_name:
            new_name = var_name
            new_names.append(new_name)
        else:
            for replace_to in replace_to_list:
                new_name = var_name
                new_name = new_name.replace(replace_from, replace_to + '_6')
                new_names.append(new_name)

        for new_name in new_names:
            new_var_to_restore = slim.get_variables_by_name(new_name)
            if len(new_var_to_restore) == 1:
                print('load ' + new_var_to_restore[0].name)
                # replaced_name_list[var_name] = new_var_to_restore[0]     # cannot use fast restore because the key-value pair should be unique
                var_value = tf.contrib.framework.load_variable(checkpoint_dir, var_name)
                sess.run(new_var_to_restore[0].assign(var_value))
