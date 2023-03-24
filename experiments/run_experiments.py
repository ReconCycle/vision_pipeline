from main import Main

if __name__ == '__main__':
    # run all experiments

    # # SIFT eval
    # Main(["--mode", "eval",
    #     "--cutoff", "0.01",
    #     "--model", "sift"])

    # # superglue eval
    # Main(["--mode", "eval",
    #     "--cutoff", "0.5",
    #     "--model", "superglue"])
    
    # # pairwise_classifier train
    # main = Main(["--mode", "train",
    #             "--model", "pairwise_classifier",
    #             "--freeze_backbone", False])
    
    # results_path = main.args.results_path

    # # pairwise_classifier eval
    # Main(["--mode", "eval",
    #     "--model", "pairwise_classifier",
    #     "--results_path", results_path])
    
    # pairwise_classifier2 train
    # main = Main(["--mode", "train",
    #              "--model", "pairwise_classifier2",
    #              "--freeze_backbone", False])

    # triplet 
    main = Main(["--mode", "train", 
                 "--model", "triplet", 
                 "--freeze_backbone", "False", 
                 "--early_stopping", "True"])