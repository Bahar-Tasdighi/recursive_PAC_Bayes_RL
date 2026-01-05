from datetime import datetime
from os import path
from matplotlib import pyplot as plt


def plot_loss(self):
        """Plot the loss curve.
        Args:
            self: An instance containing loss data and configuration.
        Returns:
            None
        """
        fig, axes = plt.subplots(5, 2, figsize=(12, 14))
        excess_pos = [x[0].detach().cpu().item() for x in self.tariningloss]  # First value (positive excess)
        excess_neg = [x[1].detach().cpu().item() for x in self.tariningloss]  # Second value (negative excess)
        kl_values = [x[2].detach().cpu().item() for x in self.tariningloss]  # Third value (KL loss)
        loss = [x[3].detach().cpu().item() for x in self.tariningloss]  # Fourth value (Total loss)
        excess = [x[4].detach().cpu().item() for x in self.tariningloss]  # Fifth value (Excess loss)

       # --- B Values & Loss History ---
        ax0 = axes[0, 0]
        ax0.plot(self.loss_history, label="Loss History", color="red")
        ax0.plot(self.B_list[:-1], label="B Values", color="green", linestyle="dashed")
        ax0.set_ylabel("Loss History & B Values")
        ax0.set_xlabel("Episode Count")
        ax0.legend(loc="upper left")

        # --- Excess Loss & Total Loss (Twin Axis) ---
        ax1 = axes[0, 1]
        ax1.plot(excess, label="Excess Loss", color="blue", linestyle="dotted")
        ax1.set_ylabel("Excess Loss", color="blue")
        ax1.set_xlabel("Gradient Update")
        ax1.legend(loc="upper left")

        ax1_twin = ax1.twinx()  # Twin axis for Total Loss
        ax1_twin.plot(loss, label="Total Loss", color="red", linestyle="solid")
        ax1_twin.set_ylabel("Total Loss", color="red")
        ax1_twin.legend(loc="upper right")

        # --- Excess Positive & Negative (Together) ---
        ax2 = axes[1, 0]
        ax2.plot(excess_pos, label="Positive Excess Loss", color="blue", linestyle="dashed")
        ax2.set_ylabel("Excess Positive")
        ax2.set_xlabel("Gradient Update")
        ax2.legend(loc="upper left")

        ax2_twin = ax2.twinx()  # Twin axis for KL Loss
        ax2_twin.plot(excess_neg, label="Negative Excess Loss", color="green", linestyle="dotted")
        ax2_twin.set_ylabel("Excess Negative", color="green")
        ax2_twin.legend(loc="upper right")


        # --- KL Loss (Separate Plot) ---
        ax3 = axes[1, 1]
        ax3.plot(kl_values, label="KL Loss", color="purple", linestyle="solid")
        ax3.set_ylabel("KL Loss", color="purple")
        ax3.set_xlabel("Gradient Update")
        ax3.legend(loc="upper left")

        
        # Portions for the remaining plots (ax1 to ax8)
        
        if self.num_portions == 6:
            portions = [0, 1, 2, 3, 4, 5]  # List of portions to loop through
        elif self.num_portions == 2:
            portions = [0, 1]  # List of portions to loop through (2 portions)
        for idx, portion in enumerate(portions):
            ax = axes[(idx // 2) + 2, idx % 2]  
            ax.plot(self.emploss_posterior[portion], label="posterior loss", color="blue")
            ax.set_ylabel(f"posterior loss (portion {portion + 1})", color="blue")
            ax.set_xlabel(f"portion {portion + 1}")
            ax.legend(loc="upper left")

            ax4 = ax.twinx()
            ax4.plot(self.emploss_prior[portion], label="prior loss", color="green")
            ax4.set_ylabel(f"prior loss (portion {portion + 1})", color="green")
            ax4.legend(loc="upper right")

            plt.tight_layout()

        plt.savefig(
            f"Ant_validationround_100000/seed_0{self.seed}/Result/Zplotslogs_{self.num_portions}.png"
        )
        plt.close() 





def NR_plot_loss(self):
        """
        Generate and save a multi-panel loss visualization plot.
        Args:
            self: An instance containing loss data and configuration.
        Returns:
            None
        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        if self.loss_type == "noninformative":
            portions = list(range(1))
        else:
            portions = list(range(2))

        # --- B Values & Loss History ---
        ax0 = axes[1, 0]
        ax0.plot(self.kl_train_list, label="kl train", color="red")
        ax0.set_ylabel("kl in training")
        ax0.set_xlabel("Episode Count")
        ax0.legend(loc="upper left")

        for idx, portion in enumerate(portions):
            row, col = divmod(idx, 2)
            ax = axes[row, col]

            if self.emploss_posterior[portion]:
                ax.plot(self.emploss_posterior[portion], label="Posterior Loss", color="blue")
                ax.set_ylabel(f"Posterior (portion {portion + 1})", color="blue")
                ax.set_xlabel(f"Steps")
                ax.legend(loc="upper left")

            ax2 = ax.twinx()
            if self.emploss_prior[portion]:
                ax2.plot(self.emploss_prior[portion], label="Prior Loss", color="green")
                ax2.set_ylabel(f"Prior (portion {portion + 1})", color="green")
                ax2.legend(loc="upper right")

        plt.tight_layout()
        plt.savefig(f"Ant_validationround_100000/seed_0{self.seed}/Result/Zplotslogs_NR_{self.loss_type}.png")
        plt.close()









def logging_table(path, model, model_type, args, loss_test, loss_train):
    """Log results in a structured table format."""
    log = {
        "number of eval episodes": args.eval_episodes,
        "KLLIST": model.kllist,
        "excesslist": model.excess,
        "B_tlist_bound": model.B_tlist,
        "laststep_invbound": model.B_T_inv, 
        "average squared loss (ground truth)": loss_test,
        "average squared loss (train data)": loss_train,
    }

    columns = ["t", "n_bound", "excess", "kl", "B_t", "posterioremploss", "E_t", "inv_b" , "Epsilon_plas", "invinf_gammab" , "Epsilon_neg", "kl_n",  "max_post_loss", "min_post_loss", "max_priorloss", "min_priorloss"]
    col_widths = [
            max(
                len(columns[j]),  # Header width
                max(len(f"{model.boundsize[i]:.5f}") if j == 1 else
                    len(f"{model.excess[i]:.5f}") if j == 2 else
                    len(f"{model.kllist[i]:.5f}") if j == 3 else
                    len(f"{model.B_tlist[i]:.5f}") if j == 4 else
                    len(f"{model.posterioremploss[i]:.5f}") if j == 5 else
                    len(f"{model.E_tlist[i][0]:.5f}") if j == 6 else
                    len(f"{model.E_tlist[i][1]:.5f}") if j == 7 else
                    len(f"{model.E_tlist[i][2]:.5f}") if j == 8 else
                    len(f"{model.E_tlist[i][3]:.5f}") if j == 9 else
                    len(f"{model.E_tlist[i][4]:.5f}") if j == 10 else
                    len(f"{model.E_tlist[i][5]:.5f}") if j == 11 else
                    len(f"{model.minmax[i][0].item():.5f}") if j == 12 else
                    len(f"{model.minmax[i][1].item():.5f}") if j == 13 else
                    len(f"{model.minmax[i][2].item():.5f}") if j == 14 else
                    len(f"{model.minmax[i][3].item():.5f}") if j == 15 else
                    len(str(i + 1))  # Row numbers
                    for i in range(args.num_portions))
            ) + 2  # Add padding for space
            for j in range(len(columns))
        ]

        # Correct separator that aligns exactly with column widths
    separator = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"

    # Writing to log file
    with open(f"{path}/Result/{model_type}_{args.num_portions}portion_.log", "w") as f:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"Log created on: {now}\n\n")
            f.write(f"epoch: { model.epochs}\n")
            f.write(f"batch_size: {model.batch_size}\n\n") 
            f.write(f"local reparametrization used: {model.local_reparametrization}\n")
            # Write metadata
            for key, value in log.items():
                f.write(f"{key}: {value}\n")

            # Write header
            f.write(separator + "\n")
            f.write("| " + " | ".join(columns[j].ljust(col_widths[j]) for j in range(len(columns))) + " |\n")
            f.write(separator + "\n")

            # Write rows with correctly aligned values
            for i in range(args.num_portions):
                row = [
                    str(i + 1).rjust(col_widths[0]),  # Row number
                    f"{model.boundsize[i]:.5f}".rjust(col_widths[1]),
                    f"{model.excess[i]:.5f}".rjust(col_widths[2]),
                    f"{model.kllist[i]:.5f}".rjust(col_widths[3]),
                    f"{model.B_tlist[i]:.5f}".rjust(col_widths[4]),
                    f"{model.posterioremploss[i]:.5f}".rjust(col_widths[5]),
                    f"{model.E_tlist[i][0]:.5f}".rjust(col_widths[6]),
                    f"{model.E_tlist[i][1]:.5f}".rjust(col_widths[7]),
                    f"{model.E_tlist[i][2]:.5f}".rjust(col_widths[8]),
                    f"{model.E_tlist[i][3]:.5f}".rjust(col_widths[9]),
                    f"{model.E_tlist[i][4]:.5f}".rjust(col_widths[10]),
                    f"{model.E_tlist[i][5]:.5f}".rjust(col_widths[11]),
                    f"{model.minmax[i][0].item():.5f}".rjust(col_widths[12]),
                    f"{model.minmax[i][1].item():.5f}".rjust(col_widths[13]),
                    f"{model.minmax[i][2].item():.5f}".rjust(col_widths[14]),
                    f"{model.minmax[i][3].item():.5f}".rjust(col_widths[15]),
                ]
                f.write("| " + " | ".join(row) + " |\n")
                f.write(separator + "\n")       


def NR_logging_table(path, model, model_type, args, loss_test, loss_train, kl_final, NR_results, NR_n_bound, local_reparametrization, learning_rate):
    """Log results in a structured table format for non-recursive case."""
    env = args.env
    seed = args.seed
    eval_episodes = args.eval_episodes
    loss_type = args.loss_type
    # test_losses = []
    # train_losses = []
    # test_losses = []
    # train_losses = []
    # kl_list = []
    # nr_bounds = []
    # n_bounds_list = []

    log = {
            "eval_episodes": eval_episodes,
            "kl": kl_final,
            "NR_type": loss_type,
            "Non-recursive invkl bound": NR_results,
            "Test loss": loss_test,
            "Train loss": loss_train,
            "_NR_n_bound": NR_n_bound
    }
    with open(f"{path}/Result/NR_{loss_type}_{model_type}_{learning_rate}.log", "w") as f:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"Log created on: {now}\n\n")
            f.write(f"local reparametrization used: {local_reparametrization}\n")
            for key, value in log.items():
                f.write(f"{key}: {value}\n")