from dptb.entrypoints.main import main_parser


def test_multi_train_parser_keeps_input_init_and_logging():
    args=main_parser().parse_args(['multi-train','s2.json','-i','s1.pth','-o','out','-lp','train.log'])
    assert (args.command,args.INPUT,args.init_model,args.output,args.log_path)==('multi-train','s2.json','s1.pth','out','train.log')


def test_train_parser_remains_available():
    args=main_parser().parse_args(['train','input.json'])
    assert args.command=='train' and args.INPUT=='input.json'
