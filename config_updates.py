from sacred.config_helpers import DynamicIngredient, CMD

def add_configs(ex):

    @ex.named_config
    def nomixup():
        use_mixup = False
        mixup_alpha = 0.3


    @ex.named_config
    def mini_train():
        # just to debug
        trainer = dict(limit_train_batches=5, limit_val_batches=5)


    @ex.named_config
    def passt():
        models = {
            "net": DynamicIngredient("models.vit.passt.model_ing")
        }
    @ex.named_config
    def dynamic_roll():
        basedataset = dict(roll=True, roll_conf=dict(axis=1, shift_range=10000)
                           )


    # extra commands

    @ex.command
    def test_loaders_train_speed():
        # test how fast data is being loaded from the data loaders.
        itr = ex.datasets.training.get_iter()
        import time
        start = time.time()
        print("hello")
        for i, b in enumerate(itr):
            if i % 20 == 0:
                print(f"{i}/{len(itr)}", end="\r")
        end = time.time()
        print("totoal time:", end - start)
        start = time.time()
        print("retry:")
        for i, b in enumerate(itr):
            if i % 20 == 0:
                print(f"{i}/{len(itr)}", end="\r")
        end = time.time()
        print("totoal time:", end - start)

