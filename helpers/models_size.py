





def count_non_zero_params(model):
    sum_params = 0
    sum_non_zero = 0
    desc = ""

    def calc_params(model):
        nonlocal desc, sum_params, sum_non_zero
        skip = ""
        if "batchnorm" in type(model).__name__.lower():
             for k,p in [("running_mean", model.running_mean), ("running_var", model.running_var)]:
                 nonzero = p[p != 0].numel()
                 total = p.numel()
                 desc += f"type {type(model).__name__}, {k},  {total}, {nonzero}, {p.dtype}, {skip} " + "\n"
                 if skip != "skip":
                     sum_params += total
                     sum_non_zero += nonzero
        for k, p in model.named_parameters(recurse=False):
            nonzero = p[p != 0].numel()
            total = p.numel()
            desc += f"type {type(model).__name__}, {k},  {total}, {nonzero}, {p.dtype}, {skip} " + "\n"
            if skip != "skip":
                sum_params += total
                sum_non_zero += nonzero

    model.apply(calc_params)
    return desc, sum_params, sum_non_zero

