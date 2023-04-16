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
    #     "--model", "superglue",
    #     "--visualise", "True"])

    # # pairwise_classifier train
    # main = Main(["--mode", "train",
    #             "--model", "pairwise_classifier",
    #             "--freeze_backbone", "False"])
    
    # results_path = main.args.results_path

    # # pairwise_classifier eval
    # Main(["--mode", "eval",
    #     "--model", "pairwise_classifier",
    #     "--results_path", results_path])
    
    # pairwise_classifier2 train
    # main = Main(["--mode", "train",
    #              "--model", "pairwise_classifier2",
    #              "--freeze_backbone", "True",
    #              "--visualise", "True"])
    
    # pairwise_classifier3 train
    main = Main(["--mode", "train",
                 "--model", "pairwise_classifier3",
                 "--cutoff", "0.9",
                 "--freeze_backbone", "True",
                 "--visualise", "True"])

    # cosine eval
    # Main(["--mode", "eval",
    #     "--cutoff", "0.9",
    #     "--model", "cosine",
    #     "--visualise", "True"])

    # Clip eval
    # Main(["--mode", "eval",
    #     "--cutoff", "0.9",
    #     "--model", "clip",
    #     "--visualise", "True"])
    

    # triplet 
    # todo: implement weight decay...
    # main = Main(["--mode", "train", 
    #              "--model", "triplet",
    #              "--cutoff", "1.0",
    #              "--weight_decay", "1e-4",
    #              "--freeze_backbone", "True", 
    #              "--early_stopping", "True",
    #              "--visualise", "True"])

    