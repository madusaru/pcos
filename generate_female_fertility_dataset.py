import pandas as pd
import numpy as np

# Number of samples
n_samples = 1000
np.random.seed(42)

# Age
age = np.random.randint(20, 45, n_samples)

# Initialize empty lists
amh = []
fsh = []
lh = []
progesterone = []
follicle_count = []
ovarian_volume = []
cycle_length = []
cycle_regular = []
fertility_label = []

for i in range(n_samples):
    # Age effect
    if age[i] < 30:
        amh_val = np.random.normal(4, 1.5)
        fsh_val = np.random.normal(6, 2)
    elif age[i] < 38:
        amh_val = np.random.normal(3, 1.5)
        fsh_val = np.random.normal(7, 2)
    else:
        amh_val = np.random.normal(1.5, 1)
        fsh_val = np.random.normal(10, 3)

    # Randomly assign PCOS pattern (~15% of samples)
    if np.random.rand() < 0.15:
        # PCOS: High LH, high AMH, high follicle count, irregular cycles
        lh_val = np.random.normal(10, 3)
        amh_val = max(amh_val, np.random.normal(6, 2))
        follicle_val = np.random.randint(20, 40)
        cycle_reg = 0
        progesterone_val = np.random.normal(5, 2)
    else:
        lh_val = np.random.normal(6, 2)
        follicle_val = np.random.randint(5, 18)
        cycle_reg = np.random.choice([1,0], p=[0.8,0.2])
        progesterone_val = np.random.normal(12, 5)

    ovarian_vol = np.random.normal(8, 3)
    cycle_len = np.random.randint(21, 40)

    # Clip realistic bounds
    amh_val = np.clip(amh_val, 0.1, 10)
    fsh_val = np.clip(fsh_val, 1, 40)
    lh_val = np.clip(lh_val, 1, 30)
    progesterone_val = np.clip(progesterone_val, 0.5, 25)
    ovarian_vol = np.clip(ovarian_vol, 3, 20)
    follicle_val = np.clip(follicle_val, 3, 40)

    # Fertility label
    if (amh_val < 1 or fsh_val > 12 or progesterone_val < 5 or cycle_reg == 0):
        label = "Infertile"
    else:
        label = "Fertile"

    # Append values
    amh.append(round(amh_val,2))
    fsh.append(round(fsh_val,2))
    lh.append(round(lh_val,2))
    progesterone.append(round(progesterone_val,2))
    follicle_count.append(follicle_val)
    ovarian_volume.append(round(ovarian_vol,1))
    cycle_length.append(cycle_len)
    cycle_regular.append(cycle_reg)
    fertility_label.append(label)

# Create DataFrame
df = pd.DataFrame({
    "Age": age,
    "AMH": amh,
    "FSH": fsh,
    "LH": lh,
    "Progesterone": progesterone,
    "Follicle_count": follicle_count,
    "Ovarian_volume": ovarian_volume,
    "Cycle_length": cycle_length,
    "Cycle_regular": cycle_regular,
    "Fertility_label": fertility_label
})

# Save to CSV
df.to_csv("female_fertility_dataset_realistic.csv", index=False)
print("Dataset generated: female_fertility_dataset_realistic.csv")
print(df.head())

