import torch
from pySankey.sankey import sankey
from wordcloud import WordCloud, STOPWORDS
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from tqdm.auto import tqdm


probe_location_to_formal_name = {
    3: "MHSA",
    6: "FFN",
    7: "Hidden State",
    9: "Final Hidden State",
}
bar_width = 0.25


def get_discrete_colors(n_colors, colormap="Spectral"):
    colors = matplotlib.cm.get_cmap("Spectral", n_colors)
    cmap = plt.get_cmap("Spectral")
    colors = [cmap(i / (n_colors - 1)) for i in range(n_colors)]
    return colors


emotions_formal = [
    "joy",
    "pride",
    "surprise",
    "trust",
    "relief",
    "neutral",
    "boredom",
    "sadness",
    "fear",
    "guilt",
    "shame",
    "disgust",
    "anger",
]
formal_emotion_to_id = {emotion: i for i, emotion in enumerate(emotions_formal)}
id_to_formal_emotion = {v: k for k, v in formal_emotion_to_id.items()}
emotion_colrs = get_discrete_colors(len(emotions_formal))[::-1]
emotion_to_color = {
    emotion: emotion_colrs[i] for i, emotion in enumerate(emotions_formal)
}


def plot_correlations(
    values,
    title,
    xticklabels,
    yticklabels,
    xlabels,
    ylabels,
    figsize,
    fontsize,
    vmax,
    save_path="",
    xtick_rotations=90,
    ytick_rotations=0,
):
    fig, ax = plt.subplots(values.shape[0], values.shape[1], figsize=figsize)

    if values.shape[0] == 1 and values.shape[1] == 1:
        ax = np.array([[ax]])

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            res = values[i, j]

            cbar = (
                True if i == values.shape[0] - 1 and j == values.shape[1] - 1 else False
            )
            sns.heatmap(
                res,
                ax=ax[i, j],
                cmap="vlag",
                cbar=cbar,
                square=True,
                vmax=vmax,
                vmin=-vmax,
                cbar_kws={"ticks": [-vmax, 0, vmax]},
            )

            xticks = np.arange(res.shape[1]) + 0.5
            yticks = np.arange(res.shape[0]) + 0.5
            ax[i, j].set_xticks(
                xticks, xticklabels, fontsize=fontsize, rotation=xtick_rotations
            )
            ax[i, j].set_yticks(
                yticks, yticklabels, fontsize=fontsize, rotation=ytick_rotations
            )

            if i == values.shape[0] - 1:
                ax[i, j].set_xlabel(xlabels[j], fontsize=fontsize)

            else:
                ax[i, j].set_xticks([])

            if j == 0:
                ax[i, j].set_ylabel(ylabels[i], fontsize=fontsize)
                ax[i, j].tick_params(axis="y", rotation=ytick_rotations)

            else:
                ax[i, j].set_yticks([])

    fig.suptitle(title, fontsize=fontsize)

    if save_path != "":
        plt.savefig(f"figs/{save_path}.pdf", bbox_inches="tight", dpi=300)
    plt.show()


def generate_correlation_heatmap(d1, d2):
    d1 = d1 / d1.norm(dim=-1, keepdim=True)
    d2 = d2 / d2.norm(dim=-1, keepdim=True)

    d1_shape = d1.shape
    d2_shape = d2.shape

    assert d1_shape[:-2] == d2_shape[:-2]

    d1 = d1.reshape(-1, d1_shape[-2], d1_shape[-1])
    d2 = d2.reshape(-1, d2_shape[-2], d2_shape[-1])

    corr = torch.einsum("ijk,ilk->ijl", d1, d2)
    corr = corr.reshape(d1_shape[:-2] + (d1_shape[-2], d2_shape[-2]))

    return corr


def load_emotion_and_appraisals_list(model_name):
    emotion_appraisal_labels = torch.load(
        f"outputs/{model_name}/emotion_appraisal_labels.pt", weights_only=False
    )

    emotion_to_id = emotion_appraisal_labels["emotion_to_id"]
    id_to_emotion = {v: k for k, v in emotion_to_id.items()}

    appraisals_to_id = emotion_appraisal_labels["appraisals_to_id"]
    id_to_appraisals = {v: k for k, v in appraisals_to_id.items()}
    appraisal_to_formal_name = {
        "other_responsblt": "other\nagency",
        "predict_event": "predictability",
        "chance_control": "situational\ncontrol",
        "self_responsblt": "self\nagency",
    }

    for app in appraisals_to_id.keys():
        if app not in appraisal_to_formal_name:
            appraisal_to_formal_name[app] = app.replace("_", "\n")

    appraisal_to_formal_name_without_newline = {
        "other_responsblt": "other-agency",
        "predict_event": "predictability",
        "chance_control": "situational-control",
        "self_responsblt": "self-agency",
    }
    for app in appraisals_to_id.keys():
        if app not in appraisal_to_formal_name_without_newline:
            appraisal_to_formal_name_without_newline[app] = app.replace("_", " ")

    appraisals = list(appraisals_to_id.keys())
    coefficients = emotion_appraisal_labels["weights"]
    biasses = emotion_appraisal_labels["bias"]

    appraisal_labels = emotion_appraisal_labels["labels"][:, 1:]

    clean_logits = torch.load(
        f"outputs/{model_name}/original_logits.pt", weights_only=False
    )

    return (
        emotion_to_id,
        id_to_emotion,
        appraisals_to_id,
        id_to_appraisals,
        appraisal_to_formal_name,
        appraisal_to_formal_name_without_newline,
        appraisals,
        coefficients,
        biasses,
        clean_logits,
        appraisal_labels,
    )


def process_open_vocab(model_name, save_prefix="", freq_threshold=30):
    [ground_truth, preds_model] = torch.load(
        f"outputs/{model_name}/{save_prefix}open_vocab_predictions.pt",
        weights_only=False,
    )
    preds_keys = list(set(preds_model))
    pred_to_freq = {k: preds_model.count(k) for k in preds_keys}
    pred_to_freq = dict(
        sorted(pred_to_freq.items(), key=lambda item: item[1], reverse=True)
    )
    top_ = list(pred_to_freq.keys())[:freq_threshold]
    filtered_preds = [" others" if p not in top_ else p for p in preds_model]

    ground_truth = [l.strip() for l in ground_truth]
    preds_model = [l.strip() for l in preds_model]
    filtered_preds = [l.strip() for l in filtered_preds]

    return ground_truth, preds_model, filtered_preds


def plot_sankey(
    d1, d2, aspect=10, fontsize=8, save_path="", title="", colorDict=emotion_to_color
):
    sankey(d1, d2, aspect=aspect, fontsize=fontsize, colorDict=colorDict)
    plt.title(title)
    if save_path != "":
        plt.savefig(f"figs/{save_path}.pdf", bbox_inches="tight", dpi=300)
    return


def plot_wordcloud(
    list_of_words, save_path="", title="", width=1500, height=800, min_font_size=10
):
    # cloud_mask = np.array(Image.open("figs/cloud.pdf"))

    text = [j.strip() for j in list_of_words]
    text = " ".join(text)

    # Create a WordCloud object
    wordcloud = WordCloud(
        width=width,
        height=height,
        background_color="white",
        stopwords=STOPWORDS,
        min_font_size=min_font_size,
        collocations=False,
        # mask=cloud_mask
    ).generate(text)

    # Display the generated image
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)

    if save_path != "":
        plt.savefig(f"figs/{save_path}.pdf", bbox_inches="tight", dpi=300)

    plt.show()


def plot_confusion_matrix(
    labels_true,
    labels_predicted,
    save_prefix="",
    emotion_to_id=None,
    save_path="confusion_matrix",
    figsize=(13, 13),
    fontsize=18,
):

    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(labels_true, labels_predicted)

    # Plot the confusion matrix
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        square=True,
        xticklabels=emotion_to_id.keys(),
        yticklabels=emotion_to_id.keys(),
        cbar=False,
        annot_kws={"size": fontsize},
    )
    plt.xlabel("Predicted Labels", fontsize=fontsize)
    plt.ylabel("True Labels", fontsize=fontsize)
    plt.xticks(rotation=90, fontsize=fontsize)
    plt.yticks(rotation=0, fontsize=fontsize)
    # plt.title('Confusion Matrix', fontsize=fontsize)
    plt.savefig(f"figs/{save_path}.pdf", bbox_inches="tight", dpi=300)
    plt.show()
    correct = cm.trace()
    print("Accuracy:", correct / cm.sum(), "Total:", cm.sum(), "Correct:", correct)


def process_emotion_probe_results(model_name, layers, locs_to_probe, tokens):
    data = torch.load(
        f"outputs/{model_name}/emotion_probing_layers_{layers}_locs_{locs_to_probe}_tokens_{tokens}.pt",
        weights_only=False,
    )

    emotion_results = [
        [
            [data[l][i][token_to_pick]["accuracy_test"] for l in layers]
            for token_to_pick in tokens
        ]
        for i in locs_to_probe
    ]

    emotion_results = torch.tensor(emotion_results)

    return emotion_results


def process_non_linear_emotion_probe_results(model_name, layers, locs_to_probe, tokens):
    data = torch.load(
        f"outputs/{model_name}/non_linaer_emotion_probing_layers_{layers}_locs_{locs_to_probe}_tokens_{tokens}.pt",
        weights_only=False,
    )

    emotion_results = [
        [
            [data[l][i][token_to_pick]["accuracy_test"] for l in layers]
            for token_to_pick in tokens
        ]
        for i in locs_to_probe
    ]

    emotion_results = torch.tensor(emotion_results)

    return emotion_results


def plot_bars(
    values_,
    figsize,
    labels,
    xticklabels,
    bar_width,
    fontsize,
    titles,
    suptitle,
    xlabel,
    ylabel,
    y_low=None,
    y_high=None,
    activate_legend=True,
    legend_loc="best",
    save_path="",
):  # values: [n_plots, n_probes, n_layers]
    n_plots = values_.shape[0]
    fig, axs = plt.subplots(1, n_plots, figsize=figsize, constrained_layout=True)
    fig.suptitle(suptitle, fontsize=fontsize)
    if n_plots == 1:
        axs = [axs]
    for j in range(n_plots):
        values = values_[j]
        ax = axs[j]
        # apply y axis grid only but push it to the background
        ax.yaxis.grid(True, alpha=0.5)
        ax.set_axisbelow(True)
        ax.xaxis.grid(False)
        # Plot each row with a separate legend entry
        for i in range(values.shape[0]):
            ax.bar(
                np.arange(values.shape[1]) + i * bar_width,
                values[i],
                width=bar_width,
                label=labels[i],
            )

        # Add labels, title, and legend
        ax.set_xlabel(xlabel, fontsize=fontsize)
        if j == 0:
            ax.set_ylabel(ylabel, fontsize=fontsize)

        ax.set_title(titles[j], fontsize=fontsize)
        xticks = np.arange(values.shape[-1]) + bar_width * (values.shape[0] - 1) / 2
        ax.set_xticks(xticks, xticklabels, fontsize=fontsize / 1.2)
        if y_low is not None and y_high is not None:
            ax.set_ylim(y_low, y_high)
        ax.tick_params(axis="y", labelsize=fontsize / 1.2)

    if activate_legend:
        axs[-1].legend(fontsize=fontsize / 1.4, loc=legend_loc)
    if save_path != "":
        plt.savefig(f"figs/{save_path}.pdf", bbox_inches="tight", dpi=300)
    # Show the plot
    plt.show()


def plot_heatmap(
    vals,
    titles,
    xticks,
    xtick_labels,
    fontsize,
    yticks,
    ytick_labels,
    yticks_rotation,
    xticks_rotation,
    suptitle=None,
    subtitle="Probe at",
    vmax=None,
    vmin=None,
    y_axis_label="Tokens",
    x_axis_label="Layers",
    cmap="magma",
    cmap_label="Accuracy",
    cmap_shrink=1.0,
    cmap_aspect=10,
    cmap_fraction=0.03,
    cmap_pad=0.02,
    cbar_yticks=None,
    cbar_ytick_labels=None,
    figsize=(20, 12),
    save_path="",
):

    if vmax is None:
        vmax = vals.max().item()
    if vmin is None:
        vmin = vals.min().item()

    if cbar_yticks is None or cbar_ytick_labels is None:
        # Format to 2 decimal places
        vmax_str = f"{vmax:.2f}"
        vmin_str = f"{vmin:.2f}"
        cbar_yticks = [vmin, vmax]
        cbar_ytick_labels = [vmin_str, vmax_str]

    fig, axes = plt.subplots(1, vals.shape[0], figsize=figsize, constrained_layout=True)

    if vals.shape[0] == 1:
        axes = [axes]
    if suptitle:
        fig.suptitle(suptitle, fontsize=fontsize)

    norm = plt.Normalize(vmax, vmin)

    for i in range(len(vals)):
        cbar = False
        sns.heatmap(
            vals[i], ax=axes[i], cmap=cmap, vmin=vmin, vmax=vmax, cbar=cbar, square=True
        )
        axes[i].set_title(titles[i], fontsize=fontsize)
        axes[i].set_xlabel(x_axis_label, fontsize=fontsize)
        if i == 0:
            axes[i].set_ylabel(y_axis_label, fontsize=fontsize)

        axes[i].set_xticks(
            xticks, xtick_labels, fontsize=fontsize / 1.2, rotation=xticks_rotation
        )
        if i == 0:
            axes[i].set_yticks(
                yticks, ytick_labels, fontsize=fontsize, rotation=yticks_rotation
            )
        else:
            axes[i].set_yticks([])
        axes[i].tick_params(axis="both", which="both", length=0)

        # axes[i].yaxis.set_tick_params(rotation=yticks_rotation)
        # axes[i].xaxis.set_tick_params(rotation=xticks_rotation)

    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=axes,
        orientation="vertical",
        shrink=cmap_shrink,
        aspect=cmap_aspect,
        fraction=cmap_fraction,
        pad=cmap_pad,
    )
    cbar.set_label(cmap_label, fontsize=fontsize)
    cbar.outline.set_color("white")

    # turn off cbar ticks
    # cbar.ax.set_yticklabels(['0', '1'])
    cbar.ax.set_yticks(cbar_yticks, cbar_ytick_labels, fontsize=fontsize / 1.2)
    cbar.ax.tick_params(labelsize=fontsize / 1.2, axis="both", which="both", length=0)

    if save_path != "":
        plt.savefig(f"figs/{save_path}.pdf", bbox_inches="tight", dpi=300)
    plt.show()


def process_zero_intervention(model_name, locs_to_probe, tokens, layers_centers, span):
    values = torch.zeros(len(locs_to_probe), len(tokens), len(layers_centers))
    clean_logits = torch.load(
        f"outputs/{model_name}/original_logits.pt", weights_only=False
    )

    for j, token_to_pick in enumerate(tokens):
        # fixing backward compatibility
        for k, loc in enumerate(locs_to_probe):
            data = torch.load(
                f"outputs/{model_name}/zero_intervention_results_layers_{layers_centers}_span_{span}_locs_{[loc]}_token_{token_to_pick}.pt",
                weights_only=False,
            )
            data = torch.stack([data[l] for l in layers_centers])
            values[k, j] = (
                data.argmax(dim=-1).eq(clean_logits.argmax(-1)).float().mean(dim=-1)
            )

    return values


def process_random_intervention(
    model_name, locs_to_probe, tokens, layers_centers, span, seed
):
    values = torch.zeros(len(locs_to_probe), len(tokens), len(layers_centers))

    clean_logits = torch.load(
        f"outputs/{model_name}/original_logits.pt", weights_only=False
    )
    # print('clean logits loaded with shape:', clean_logits.shape)
    # print('--------------------------------------------')
    for j, token_to_pick in enumerate(tokens):
        for k, loc in enumerate(locs_to_probe):
            data = torch.load(
                f"outputs/{model_name}/random_intervention_results_layers_{layers_centers}_span_{span}_locs_{[loc]}_token_{token_to_pick}_seed_{seed}.pt",
                weights_only=False,
            )
            # print('data loaded with shape:', data[layers_centers[0]].shape)
            if clean_logits.shape[0] != data[layers_centers[0]].shape[0]:
                # print('loading clean logits from new file')
                clean_logits = torch.load(
                    f"outputs/{model_name}/original_logits_new.pt", weights_only=False
                )  # a little dirty code because the transformer library updated Olmo model behavior during our experiments
                # print('clean logits loaded with shape:', clean_logits.shape)
            data = torch.stack([data[l] for l in layers_centers])
            values[k, j] = (
                data.argmax(dim=-1).eq(clean_logits.argmax(-1)).float().mean(dim=-1)
            )
            # print('--------------------------------------------')

    return values


def process_activation_patching(
    model_name, num_exps, locs_to_probe, tokens, layers_centers, span
):
    values = torch.zeros((len(locs_to_probe), len(tokens), len(layers_centers), 3))

    for l, loc in tqdm(enumerate(locs_to_probe), total=len(locs_to_probe)):
        for t, token_to_pick in enumerate(tokens):
            data = torch.load(
                f"outputs/{model_name}/patching_results_layers_{layers_centers}_locs_{[loc]}_span_{span}_token_{[token_to_pick]}.pt",
                weights_only=False,
            )

            for k, lc in enumerate(layers_centers):
                source_logits_clean = [
                    data[lc][i]["source_clean_logits"] for i in range(num_exps)
                ]
                patched_logits = [
                    data[lc][i]["patched_logits"] for i in range(num_exps)
                ]
                target_logits_clean = [
                    data[lc][i]["target_clean_logits"] for i in range(num_exps)
                ]

                source_labels_clean = torch.stack(source_logits_clean).argmax(dim=-1)
                patched_labels = torch.stack(patched_logits).argmax(dim=-1)
                target_labels_clean = torch.stack(target_logits_clean).argmax(dim=-1)

                values[l, t, k, 0] = (
                    (source_labels_clean == patched_labels).float().mean()
                )
                values[l, t, k, 1] = (
                    (patched_labels == target_labels_clean).float().mean()
                )
                values[l, t, k, 2] = 1.0 - values[l, t, k, 0] - values[l, t, k, 1]

    # torch.save(values,
    #            f'outputs/{model_name}/processed_patching_results_layers_{layers_centers}_locs_{locs_to_probe}_span_{span}_tokens_{tokens}.pt')

    return values


def plot_cumulative_bar(
    values,
    titles,
    bar_width,
    figsize,
    fontsize,
    xtick_labels_,
    x_low,
    x_high,
    y_low,
    y_high,
    colors,
    labels,
    legend_loc,
    legend_index,
    save_path="",
):
    fig, axs = plt.subplots(1, len(values), figsize=figsize, constrained_layout=True, squeeze=False)
    for j in range(len(values)):
        xtick_labels = xtick_labels_[j]
        ax = axs[0][j]
        for i in range(values[j].shape[-1]):
            ax.bar(
                xtick_labels,
                values[j][..., i],
                width=bar_width,
                label=labels[i],
                color=colors[i],
                bottom=values[j][..., :i].sum(axis=-1),
            )

        ax.set_xlabel("Layers", fontsize=fontsize)

        ax.set_title(titles[j], fontsize=fontsize)
        ax.set_ylim(y_low, y_high)
        ax.set_xlim(x_low, x_high)
        ax.set_xticks(xtick_labels, xtick_labels, fontsize=fontsize / 1.2)
        ax.tick_params(axis="x", labelsize=fontsize / 1.2)
        if j == 0:
            ax.set_ylabel("Accuracy", fontsize=fontsize)
            ax.tick_params(axis="y", labelsize=fontsize / 1.2)
        else:
            ax.set_yticks([])
        if j == legend_index:
            ax.legend(loc=legend_loc, fontsize=fontsize / 1.4)
        ax.tick_params(axis="y", labelsize=fontsize / 1.2)
    if save_path != "":
        plt.savefig(f"figs/{save_path}.pdf", bbox_inches="tight", dpi=300)
    plt.show()


def sort_two_emotion_lists_based_on_the_first_one(first_list, second_list, reference):
    reference_to_id = {r: i for i, r in enumerate(reference)}
    id_to_reference = {i: r for i, r in enumerate(reference)}
    first = [reference_to_id[f] for f in first_list]

    first_sorted, second_sorted = zip(*sorted(zip(first, second_list)))

    first_sorted = [id_to_reference[f] for f in first_sorted]

    return first_sorted, second_sorted


def normalize_row_wise_with_nan_mask(values):
    values = values.clone()
    non_nan_mask = ~torch.isnan(values)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if non_nan_mask[i, j].sum() == 0:
                continue
            max = values[i, j, non_nan_mask[i, j]].max(-1, keepdim=True).values
            min = values[i, j, non_nan_mask[i, j]].min(-1, keepdim=True).values
            values[i, j, non_nan_mask[i, j]] = (
                values[i, j, non_nan_mask[i, j]] - min
            ) / (max - min)
    return values
