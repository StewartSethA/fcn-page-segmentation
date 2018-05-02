def build_model(args):
    if args.framework.lower() == "keras":
        from keras_models import build_model as model_builder
    elif args.framework.lower() == "tensorflow":
        from tensorflow_models import build_model as model_builder
    elif args.framework.lower() == "pytorch" or True:
        raise "Invalid deep learning framework or not implemented:", args.framework
    return model_builder(args)
