def arg_test(*args):
    print('args:')
    print(args[0])
    print('arg')
    for arg in args[0]:        print(arg)


metrics_list = ['Dice', 'Jaccard', 'Hausdorff']

arg_test(*metrics_list)