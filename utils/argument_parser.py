
def args_parse():
    parser = argparse.ArgumentParser(description='Supply naive CNN with hyper-parameter configuration & options')
    
    # The parameters for base model compilation
    compile_env = parser.add_argument_group(title="Base Model Compilation")
    compile_env.add_argument('--optimizer', '-o', default='adam', type=str, help='Choice of an optimzer to use when compiling the model during training and validation')
    compile_env.add_argument('--learning-rate', '-lr', default=0.01, type=float, help='Select a learning rate for the model to use when updating weights during training')
    compile_env.add_argument('--loss-fn', '-lf', default='categorical_crossentropy', type=str, help='Choise of a loss function to minimise and use during update step')
    compile_env.add_argument('--metrics', '-me', default='categorical_accuracy', type=str, help='Metric to use for model compilation')
    compile_env.add_argument('--momentum', '-mo', default=0.9, type=float, help='Momentum value to use in the special case of a Stochastic Gradient Descent with weight decay')
    compile_env.add_argument('--weight-decay', '-wd', default=1, type=float, help='Weight decay value to use in the special case of a Stocastic Gradient Descent with weight decay')

    # The parameters for our training dataset
    train_ds_env = parser.add_argument_group(title="Parameters for Training Dataset")
    train_ds_env.add_argument('--batch-size', '-b', default=32, type=int, help='Number of data instances per batch when multi-batching')
    train_ds_env.add_argument('--num-epochs', '-e', default=200, type=int, help='Number of epochs to use during training')
    train_ds_env.add_argumment'--valid-split', '-v', default=0.20, type=float, help='Validation split ratio')
    train_ds_env.add_argument('--gpu-enable', '-gb', default=True, type=bool, help='Enable training using GPU')
    train_ds_env.add_argument('--gpu-number', '-gn', default=True, type=bool, help='Enable training using GPU')
    #train_ds_env.add_argument('--data-augment', '-da', default=False, type=bool, help='Option for including data augmentation for the training and validation datasets.')
    
    # The argument group for Naive NN Training
    naive_nn_env = parser.add_argument_group(title="Parameters for Naive NN Model")
    naive_nn_env.add_argument('--', '-', default=, type=, help=)
    naive_nn_env.add_argument('--', '-', default=, type=, help=)

    # The additional argument group for iCaRL Training
    train_icarl_env = parser.add_argument_group(title="Additional Parameters for Training iCaRL")
    train_icarl_env.add_argument('--K', '-k', default=2000, type=int, help="Upper limit for total number of exemplar elements")
    train_icarl_env.add_argument('--alpha', '-al', default=1.00, type=float, help="Alpha constant for distillation loss term in iCarl loss computation")
    train_icarl_env.add_argument('--beta', '-be', default=1.00, type=float, help="Beta constant for classifiaction loss term in iCarl loss computation")
    #train_icarl_env.add_argument('', '', )

    # TO-DO: The argument group for EWC Training

    # The parameters for the incremental comparator evaluation tool
    incr_comp_env = parser.add_argument(title="Parameters for Incremental Comparison Tool")
    incr_comp_env.add_argument()

    args = parser.parse_args()
    return args


#def unit_test_parser():

    