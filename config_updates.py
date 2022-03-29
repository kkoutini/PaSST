from sacred.config_helpers import DynamicIngredient, CMD


def add_configs(ex):
    '''
    This functions add generic configuration for the experiments, such as mix-up, architectures, etc...
    @param ex: Ba3l Experiment
    @return:
    '''

    @ex.named_config
    def nomixup():
        'Don\'t apply mix-up (spectrogram level).'
        use_mixup = False
        mixup_alpha = 0.3

    @ex.named_config
    def mixup():
        ' Apply mix-up (spectrogram level).'
        use_mixup = True
        mixup_alpha = 0.3

    @ex.named_config
    def mini_train():
        'limit training/validation to 5 batches for debbuging.'
        trainer = dict(limit_train_batches=5, limit_val_batches=5)

    @ex.named_config
    def passt():
        'use PaSST model'
        models = {
            "net": DynamicIngredient("models.passt.model_ing")
        }

    @ex.named_config
    def passt_s_20sec():
        'use PaSST model pretrained on Audioset (with SWA) ap=476; time encodings for up to 20 seconds'
        # python ex_audioset.py evaluate_only with passt_s_ap476
        models = {
            "net": DynamicIngredient("models.passt.model_ing", arch="passt_s_f128_20sec_p16_s10_ap474", fstride=10,
                                     tstride=10, input_tdim=2000)
        }
        basedataset = dict(clip_length=20)

    @ex.named_config
    def passt_s_30sec():
        'use PaSST model pretrained on Audioset (with SWA) ap=476; time encodings for up to 30 seconds'
        # python ex_audioset.py evaluate_only with passt_s_ap476
        models = {
            "net": DynamicIngredient("models.passt.model_ing", arch="passt_s_f128_30sec_p16_s10_ap473", fstride=10,
                                     tstride=10, input_tdim=3000)
        }
        basedataset = dict(clip_length=20)

    @ex.named_config
    def passt_s_ap476():
        'use PaSST model pretrained on Audioset (with SWA) ap=476'
        # python ex_audioset.py evaluate_only with passt_s_ap476
        models = {
            "net": DynamicIngredient("models.passt.model_ing", arch="passt_s_swa_p16_128_ap476", fstride=10,
                                     tstride=10)
        }

    @ex.named_config
    def passt_s_ap4763():
        'use PaSST model pretrained on Audioset (with SWA) ap=4763'
        # test with: python ex_audioset.py evaluate_only with passt_s_ap4763
        models = {
            "net": DynamicIngredient("models.passt.model_ing", arch="passt_s_swa_p16_128_ap4763", fstride=10,
                                     tstride=10)
        }

    @ex.named_config
    def passt_s_ap472():
        'use PaSST model pretrained on Audioset (no SWA) ap=472'
        # test with: python ex_audioset.py evaluate_only with passt_s_ap472
        models = {
            "net": DynamicIngredient("models.passt.model_ing", arch="passt_s_p16_128_ap472", fstride=10,
                                     tstride=10)
        }

    @ex.named_config
    def passt_s_p16_s16_128_ap468():
        'use PaSST model pretrained on Audioset (no SWA) ap=468 NO overlap'
        # test with: python ex_audioset.py evaluate_only with passt_s_p16_s16_128_ap468
        models = {
            "net": DynamicIngredient("models.passt.model_ing", arch="passt_s_p16_s16_128_ap468", fstride=16,
                                     tstride=16)
        }

    @ex.named_config
    def passt_s_swa_p16_s16_128_ap473():
        'use PaSST model pretrained on Audioset (SWA) ap=473 NO overlap'
        # test with: python ex_audioset.py evaluate_only with passt_s_swa_p16_s16_128_ap473
        models = {
            "net": DynamicIngredient("models.passt.model_ing", arch="passt_s_swa_p16_s16_128_ap473", fstride=16,
                                     tstride=16)
        }

    @ex.named_config
    def passt_s_swa_p16_s14_128_ap471():
        'use PaSST model pretrained on Audioset stride=14 (SWA) ap=471 '
        # test with: python ex_audioset.py evaluate_only with passt_s_swa_p16_s14_128_ap471
        models = {
            "net": DynamicIngredient("models.passt.model_ing", arch="passt_s_swa_p16_s14_128_ap471", fstride=14,
                                     tstride=14)
        }

    @ex.named_config
    def passt_s_p16_s14_128_ap469():
        'use PaSST model pretrained on Audioset stride=14 (No SWA) ap=469 '
        # test with: python ex_audioset.py evaluate_only with passt_s_p16_s14_128_ap469
        models = {
            "net": DynamicIngredient("models.passt.model_ing", arch="passt_s_p16_s14_128_ap469", fstride=14,
                                     tstride=14)
        }

    @ex.named_config
    def passt_s_swa_p16_s12_128_ap473():
        'use PaSST model pretrained on Audioset stride=12 (SWA) ap=473 '
        # test with: python ex_audioset.py evaluate_only with passt_s_swa_p16_s12_128_ap473
        models = {
            "net": DynamicIngredient("models.passt.model_ing", arch="passt_s_swa_p16_s12_128_ap473", fstride=12,
                                     tstride=12)
        }

    @ex.named_config
    def passt_s_p16_s12_128_ap470():
        'use PaSST model pretrained on Audioset stride=12 (No SWA) ap=4670 '
        # test with: python ex_audioset.py evaluate_only with passt_s_p16_s12_128_ap470
        models = {
            "net": DynamicIngredient("models.passt.model_ing", arch="passt_s_p16_s12_128_ap470", fstride=12,
                                     tstride=12)
        }

    @ex.named_config
    def ensemble_s10():
        'use ensemble of PaSST models pretrained on Audioset  with S10 mAP=.4864'
        # test with: python ex_audioset.py evaluate_only with  trainer.precision=16 ensemble_s10
        models = {
            "net": DynamicIngredient("models.passt.model_ing", arch="ensemble_s10", fstride=None,
                                     tstride=None, instance_cmd="get_ensemble_model",
                                     # don't call get_model but rather get_ensemble_model
                                     arch_list=[
                                         ("passt_s_swa_p16_128_ap476", 10, 10),
                                         ("passt_s_swa_p16_128_ap4761", 10, 10),
                                         ("passt_s_p16_128_ap472", 10, 10),
                                     ]
                                     )
        }

    @ex.named_config
    def ensemble_many():
        'use ensemble of PaSST models pretrained on Audioset  with different strides mAP=.4956'
        # test with: python ex_audioset.py evaluate_only with  trainer.precision=16 ensemble_many
        models = {
            "net": DynamicIngredient("models.passt.model_ing", arch="ensemble_many", fstride=None,
                                     tstride=None, instance_cmd="get_ensemble_model",
                                     # don't call get_model but rather get_ensemble_model
                                     arch_list=[
                                         ("passt_s_swa_p16_128_ap476", 10, 10),
                                         ("passt_s_swa_p16_128_ap4761", 10, 10),
                                         ("passt_s_p16_128_ap472", 10, 10),
                                         ("passt_s_p16_s12_128_ap470", 12, 12),
                                         ("passt_s_swa_p16_s12_128_ap473", 12, 12),
                                         ("passt_s_p16_s14_128_ap469", 14, 14),
                                         ("passt_s_swa_p16_s14_128_ap471", 14, 14),
                                         ("passt_s_swa_p16_s16_128_ap473", 16, 16),
                                         ("passt_s_p16_s16_128_ap468", 16, 16),
                                     ]
                                     )
        }

    @ex.named_config
    def ensemble_4():
        'use ensemble of PaSST models pretrained on Audioset  with different strides mAP=.4926'
        # test with: python ex_audioset.py evaluate_only with  trainer.precision=16 ensemble_many
        models = {
            "net": DynamicIngredient("models.passt.model_ing", arch="ensemble_many", fstride=None,
                                     tstride=None, instance_cmd="get_ensemble_model",
                                     # don't call get_model but rather get_ensemble_model
                                     arch_list=[
                                         ("passt_s_swa_p16_128_ap476", 10, 10),
                                         ("passt_s_swa_p16_s12_128_ap473", 12, 12),
                                         ("passt_s_swa_p16_s14_128_ap471", 14, 14),
                                         ("passt_s_swa_p16_s16_128_ap473", 16, 16),
                                     ]
                                     )
        }

    @ex.named_config
    def ensemble_5():
        'use ensemble of PaSST models pretrained on Audioset  with different strides mAP=.49459'
        # test with: python ex_audioset.py evaluate_only with  trainer.precision=16 ensemble_many
        models = {
            "net": DynamicIngredient("models.passt.model_ing", arch="ensemble_many", fstride=None,
                                     tstride=None, instance_cmd="get_ensemble_model",
                                     # don't call get_model but rather get_ensemble_model
                                     arch_list=[
                                         ("passt_s_swa_p16_128_ap476", 10, 10),
                                         ("passt_s_swa_p16_128_ap4761", 10, 10),
                                         ("passt_s_swa_p16_s12_128_ap473", 12, 12),
                                         ("passt_s_swa_p16_s14_128_ap471", 14, 14),
                                         ("passt_s_swa_p16_s16_128_ap473", 16, 16),
                                     ]
                                     )
        }

    @ex.named_config
    def ensemble_s16_14():
        'use ensemble of two PaSST models pretrained on Audioset  with stride 16 and 14 mAP=.48579'
        # test with: python ex_audioset.py evaluate_only with  trainer.precision=16 ensemble_s16_14
        models = {
            "net": DynamicIngredient("models.passt.model_ing", arch="ensemble_s16", fstride=None,
                                     tstride=None, instance_cmd="get_ensemble_model",
                                     # don't call get_model but rather get_ensemble_model
                                     arch_list=[
                                         ("passt_s_swa_p16_s14_128_ap471", 14, 14),
                                         ("passt_s_swa_p16_s16_128_ap473", 16, 16),
                                     ]
                                     )
        }

    @ex.named_config
    def dynamic_roll():
        # dynamically roll the spectrograms/waveforms
        # updates the dataset config
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
