from datetime import datetime
import os

from plotting import NR_logging_table, logging_table
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import argparse
from evaluator import NR_PerformanceEval, PerformanceEval

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="ant")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--baseline_category", type=str, default="nonrecursive", choices=["nonrecursive", "recursive"])
    parser.add_argument("--eval_episodes", type=int, default=100)
    parser.add_argument("--model_type", type=str, default="fullvb", choices=["fullvb", "lastlayervb"])
    parser.add_argument("--num_portions", type=int, default=2, choices=[2, 6], help="Number of portions to divide the training episodes into (2 or 6).")
    parser.add_argument("--loss_type", type=str, default="informative", choices=["informative", "noninformative"])

    args, _ = parser.parse_known_args()

    env_limits = {"ant": 500000, "walker2d": 300000, "cheetah": 900000}
    limit = env_limits.get(args.env, 300000)
    path = f"Ant_validationround_100000/seed_0{args.seed}"
    model_type = args.model_type
    
    if args.baseline_category == "nonrecursive":
        # Run non-recursive case
        model = NR_PerformanceEval(args.env, args.model_type, args.loss_type, args.seed, args.eval_episodes, 200, limit)
        model.calculate_bound()
        loss_test, loss_train = model.predict()
        kl_final = model.kl_final
        NR_results = model.NR_results
        NR_n_bound = model._NR_n_bound
        local_reparametrization = model.local_reparametrization
        learning_rate = model.learning_rate

        print(f"Test Loss: {loss_test:.4f}")
        NR_logging_table(path=path, model=model, model_type=args.model_type, args=args, loss_test=loss_test,
                          loss_train=loss_train, kl_final=kl_final, NR_results=NR_results, NR_n_bound=NR_n_bound, 
                          local_reparametrization=local_reparametrization, learning_rate=learning_rate)

    elif args.baseline_category == "recursive": 
        # Run recursive case
        model = PerformanceEval(args.env, args.model_type, args.seed, args.eval_episodes, 200, limit, args.num_portions)
        loss_test, loss_train = model.predict()
        print(f"Test Loss: {loss_test:.4f}")
        logging_table(path=path, model=model, model_type=model_type, args=args, loss_test=loss_test, loss_train=loss_train)



if __name__ == "__main__":
    main()