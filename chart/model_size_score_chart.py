import matplotlib.pyplot as plt
import pandas as pd

data = {
    'Model': ['Nano (n)', 'Small (s)', 'Medium (m)', 'Large (l)', 'XL (x)'],
    'mAP50_95': [0.2666, 0.4721, 0.5732, 0.5442, 0.6134],
    'FPS': [85.23, 69.23, 48.80, 36.22, 26.30]
}
df = pd.DataFrame(data)

mAP_min, mAP_max = df['mAP50_95'].min(), df['mAP50_95'].max()
FPS_min, FPS_max = df['FPS'].min(), df['FPS'].max()

df['mAP_norm'] = (df['mAP50_95'] - mAP_min) / (mAP_max - mAP_min)
df['FPS_norm'] = (df['FPS'] - FPS_min) / (FPS_max - FPS_min)
df['Score'] = (0.5 * df['mAP_norm'] + 0.5 * df['FPS_norm'])

fig, ax = plt.subplots(figsize=(9, 6))

bars = ax.bar(
    df['Model'], 
    df['Score'], 
    color='steelblue', 
    edgecolor='black',
    linewidth=1.2,
    width=0.6
)

bars[1].set_color('mediumseagreen') # Small
bars[1].set_edgecolor('black')
bars[2].set_color('mediumseagreen') # Medium
bars[2].set_edgecolor('black')

for bar in bars:
    yval = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width()/2, 
        yval + 0.005,
        f'{yval:.2f}', 
        ha='center', 
        va='bottom', 
        fontweight='bold',
        fontsize=11
    )

ax.set_xlabel('Modellgröße', fontsize=13, labelpad=10)
ax.set_ylabel('Güte (normalisiert)', fontsize=13, labelpad=10)

ax.set_ylim(0.3, 0.7)

plt.tight_layout()
plt.savefig('combined_score_barchart.png', dpi=300, bbox_inches='tight')

plt.show()