class SetAdapterScale:
    def __enter__(self, model, scale):
        print("Enter Context Manager")
        print("[INFO] MODEL")
        print(model)
        print("[INFO] SCALE")
        print(scale)
        
        # 1. Check whether scaling is prohibited on model
        print("Checking ...")

        # 2. If scaling is allowed, scale the weights
        print("Scaling ...")

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Exit Context Manager")