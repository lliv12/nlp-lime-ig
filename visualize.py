import matplotlib.pyplot as plt
import numpy as np

'''
Generate a bar chart showing the model accuracy over multiple classes; save as .png image

file_loc:  path to save the image in (including name)
correct: tensor(L)  (tensor where each element is #correct for class [i])  (L: #labels)
labels:  tensor(L)  (tensor where each element is #labels encountered for class [i])  (L: #labels)
class_labels:  list[L]  (list of labels for each class)  (L: #labels)
'''
def bar_chart(file_loc, correct, labels, class_labels):
    x_ticks = np.arange(len(class_labels))  # x-axis values for each class
    color_mapping = {0: 'red', 1: 'orange', 2: 'yellow', 3: 'lawngreen', 4: 'green'}
    colors = [color_mapping[i] for i in range(len(class_labels))]

    precision = correct / labels
    recall = correct / labels.sum()
    f1_score = 2 * (precision * recall) / (precision + recall)

    # bar chart
    fig = plt.figure(figsize=(9, 12))
    gs = fig.add_gridspec(2, 1, height_ratios=[4, 1])

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(class_labels)
    ax1.bar(x_ticks, labels, color='lightgrey', label='Total')
    ax1.bar(x_ticks, correct, color=colors, label='Correct')
    ax1.tick_params(axis='both', labelsize=15)
    ax1.set_ylabel('Count', fontsize=22, labelpad=15)
    ax1.set_title('Correct Guesses vs Total Examples by Class', fontsize=22, y=1.05)
    
    perc_correct = precision * 100
    for i, v in enumerate(perc_correct):
        ax1.text(i, labels[i], f"{v:.1f}%", ha='center', va='bottom', fontsize=15)
        ratio_text = f"{int(correct[i])} / {int(labels[i])}"
        ax1.text(i, -0.08*(labels.max()), ratio_text, ha='center', va='top', fontsize=15)

    # table
    ax2 = fig.add_subplot(gs[1, 0])
    table_data = list(zip(class_labels, np.around(precision.numpy(), 2), np.around(recall.numpy(), 2), np.around(f1_score.numpy(), 2)))
    table_cols = ['Class', 'Precision', 'Recall', 'F1 Score']
    table_colors = [[colors[i], "w", "w", "w"] for i in range(len(class_labels))]
    table = ax2.table(cellText=table_data, colLabels=table_cols, cellColours=table_colors, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(13)
    table.scale(1, 1.5)
    ax2.axis('off')

    plt.subplots_adjust(hspace=0.3)

    plt.savefig(file_loc, format='png')

    plt.close()


'''
Generate a confusion matrix, save as .png image

file_loc:  path to save the image in (including name)
mat:  tensor(L, L)  (tensor where each cell is #times class [j] was predicted for ground truth class [i])  (L: #labels)
class_labels:  list[L]  (list of labels for each class)  (L: #labels)
'''
def confusion_matrix(file_loc, mat, class_labels):
    mat = mat / mat.sum(axis=1)[:, np.newaxis]
    # Plot the confusion matrix
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(mat, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    # Set up axes
    ticks = range(mat.shape[0])
    ax.set_xticks(ticks)
    ax.set_xticklabels(class_labels)
    ax.xaxis.set_ticks_position('bottom')
    ax.set_yticks(ticks)
    ax.set_yticklabels(class_labels)

    ax.set_xlabel('Predicted Rating', fontsize=17, labelpad=6)
    ax.set_ylabel('Ground Truth Rating', fontsize=17, labelpad=12)
    ax.set_title('Confusion Matrix', fontsize=20, y=1.05)

    # Add labels inside the cells
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f'{mat[i, j].item():.1%}', ha='center', va='center')

    # Show the plot
    plt.savefig(file_loc, format='png')
    plt.close()