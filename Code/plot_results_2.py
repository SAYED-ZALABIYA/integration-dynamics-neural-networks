import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 15,
    'legend.fontsize': 11,
    'figure.figsize': (10, 6)
})

# ===================== LOAD =====================
def load_data():
    df_mlp = pd.read_csv("results_enhanced.csv")
    df_rnn = pd.read_csv("rnn_iit_results.csv")

    if 'seed' in df_mlp.columns:
        df_mlp = df_mlp.groupby('epoch').mean(numeric_only=True).reset_index()

    return df_mlp, df_rnn


# ===================== FIG 1 =====================
def plot_figure_1(df_mlp):
    fig, ax1 = plt.subplots()

    ax1.plot(df_mlp['epoch'], df_mlp['acc'],
             color='#b22222', linewidth=2.5, label='Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1.05)

    ax2 = ax1.twinx()
    ax2.plot(df_mlp['epoch'], df_mlp['tc_sum'],
             color='#1f77b4', linestyle='--', linewidth=2.5, label='TC')
    ax2.set_ylabel('Statistical Integration (TC)')

    plt.title('Training Dynamics: Accuracy vs Statistical Integration (MLP)')
    fig.tight_layout()
    plt.savefig('Figure_1.png', dpi=300)
    plt.close()


# ===================== FIG 2 =====================
def plot_figure_2(df_mlp, df_rnn):
    plt.figure()

    plt.plot(df_mlp['epoch'], df_mlp['tc_sum'],
             label='MLP', linestyle='--', linewidth=2.5, color='#1f77b4')

    plt.plot(df_rnn['epoch'], df_rnn['tc_sum'],
             label='RNN', linewidth=2.5, color='#ff7f0e')

    plt.xlabel('Epoch')
    plt.ylabel('Statistical Integration (TC)')
    plt.title('Statistical Integration Dynamics: MLP vs RNN')

    plt.legend()
    plt.tight_layout()
    plt.savefig('Figure_2.png', dpi=300)
    plt.close()


# ===================== FIG 3 =====================
def plot_figure_3(df_mlp, df_rnn):
    plt.figure()

    # Scatter
    plt.scatter(df_mlp['tc_sum'], df_mlp['acc'],
                color='#1f77b4', alpha=0.7, s=60, label='MLP')

    plt.scatter(df_rnn['tc_sum'], df_rnn['accuracy'],
                color='#d62728', alpha=0.7, s=60, marker='^', label='RNN')

    # Correlation
    r_mlp = np.corrcoef(df_mlp['tc_sum'], df_mlp['acc'])[0, 1]
    r_rnn = np.corrcoef(df_rnn['tc_sum'], df_rnn['accuracy'])[0, 1]

    # Trend lines
    z_mlp = np.polyfit(df_mlp['tc_sum'], df_mlp['acc'], 1)
    p_mlp = np.poly1d(z_mlp)
    plt.plot(df_mlp['tc_sum'], p_mlp(df_mlp['tc_sum']),
             linestyle='--', color='#1f77b4', alpha=0.5)

    z_rnn = np.polyfit(df_rnn['tc_sum'], df_rnn['accuracy'], 1)
    p_rnn = np.poly1d(z_rnn)
    plt.plot(df_rnn['tc_sum'], p_rnn(df_rnn['tc_sum']),
             linestyle='--', color='#d62728', alpha=0.5)

    # Annotation
    plt.text(min(df_mlp['tc_sum']), max(df_mlp['acc']),
             f"MLP r = {r_mlp:.2f}", fontsize=11, color='#1f77b4')

    plt.text(min(df_rnn['tc_sum']), max(df_rnn['accuracy']) - 0.03,
             f"RNN r = {r_rnn:.2f}", fontsize=11, color='#d62728')

    plt.xlabel('Statistical Integration (TC)')
    plt.ylabel('Accuracy')
    plt.title('Inverse Relationship between Accuracy and Statistical Integration')

    plt.legend()
    plt.tight_layout()
    plt.savefig('Figure_3.png', dpi=300)
    plt.close()


# ===================== FIG 4 =====================
def plot_figure_4(df_mlp):
    layer_cols = [c for c in df_mlp.columns if c.startswith('tc_l')]
    if not layer_cols:
        return

    plt.figure()

    for col in layer_cols:
        plt.plot(df_mlp['epoch'], df_mlp[col], linewidth=2)

    plt.xlabel('Epoch')
    plt.ylabel('Layer-wise Statistical Integration (TC)')
    plt.title('Layer-wise Decorrelation during Training (MLP)')

    plt.tight_layout()
    plt.savefig('Figure_4.png', dpi=300)
    plt.close()


# ===================== FIG 5 =====================
def plot_figure_5(df_rnn):
    step_cols = [c for c in df_rnn.columns if c.startswith('tc_step')]
    if not step_cols:
        return

    plt.figure()

    for i, col in enumerate(step_cols):
        plt.plot(df_rnn['epoch'], df_rnn[col],
                 label=f'Time step {i+1}', linewidth=2)

    plt.xlabel('Epoch')
    plt.ylabel('Temporal Statistical Integration (TC)')
    plt.title('Temporal Integration Dynamics in RNN')

    plt.legend()
    plt.tight_layout()
    plt.savefig('Figure_5.png', dpi=300)
    plt.close()


# ===================== FIG 6 =====================
def plot_figure_6(df_mlp, df_rnn):
    values = [
        df_mlp['tc_sum'].iloc[0],
        df_mlp['tc_sum'].iloc[-1],
        df_rnn['tc_sum'].iloc[0],
        df_rnn['tc_sum'].iloc[-1]
    ]

    labels = ['MLP Start', 'MLP End', 'RNN Start', 'RNN End']

    plt.figure()
    bars = plt.bar(labels, values,
                   color=['#aec7e8', '#1f77b4', '#ff9896', '#d62728'],
                   edgecolor='black')

    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2,
                 h, f'{h:.1f}', ha='center', va='bottom')

    plt.ylabel('Statistical Integration (TC)')
    plt.title('Change in Statistical Integration (Start vs End)')

    plt.tight_layout()
    plt.savefig('Figure_6.png', dpi=300)
    plt.close()


# ===================== RUN =====================
if __name__ == "__main__":
    df_mlp, df_rnn = load_data()

    if 'accuracy' in df_mlp.columns:
        df_mlp.rename(columns={'accuracy': 'acc'}, inplace=True)

    if 'acc' in df_rnn.columns:
        df_rnn.rename(columns={'acc': 'accuracy'}, inplace=True)

    plot_figure_1(df_mlp)
    plot_figure_2(df_mlp, df_rnn)
    plot_figure_3(df_mlp, df_rnn)
    plot_figure_4(df_mlp)
    plot_figure_5(df_rnn)
    plot_figure_6(df_mlp, df_rnn)

    print("a7la msa 3ly altrmsa :)")