import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file
df = pd.read_csv('Q2_20230202_majority.csv')

# Count the occurrences of each class
class_counts = df['label_true'].value_counts()

print("Class Distribution Analysis")
print("=" * 50)
print(f"Total number of samples: {len(df)}")
print("\nClass counts:")
for class_name, count in class_counts.items():
    percentage = (count / len(df)) * 100
    print(f"{class_name}: {count} ({percentage:.2f}%)")

print("\n" + "=" * 50)

# Check for class imbalance
total_samples = len(df)
imbalance_threshold = 0.1  # 10% threshold for considering imbalance

print("Class Imbalance Analysis:")
print("=" * 50)

for class_name, count in class_counts.items():
    percentage = (count / total_samples) * 100
    if percentage < imbalance_threshold * 100:
        print(f"⚠️  {class_name} is underrepresented: {percentage:.2f}% (less than {imbalance_threshold*100}%)")
    elif percentage > (1 - imbalance_threshold) * 100:
        print(f"⚠️  {class_name} is overrepresented: {percentage:.2f}% (more than {(1-imbalance_threshold)*100}%)")
    else:
        print(f"✅ {class_name} is reasonably balanced: {percentage:.2f}%")

# Calculate imbalance metrics
max_count = class_counts.max()
min_count = class_counts.min()
imbalance_ratio = max_count / min_count

print(f"\nImbalance Ratio (max/min): {imbalance_ratio:.2f}")
if imbalance_ratio > 2:
    print("⚠️  Significant class imbalance detected (ratio > 2)")
elif imbalance_ratio > 1.5:
    print("⚠️  Moderate class imbalance detected (ratio > 1.5)")
else:
    print("✅ Classes are reasonably balanced")

# Create a bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x=class_counts.index, y=class_counts.values)
plt.title('Class Distribution in Dataset')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('class_distribution.png')
plt.show()

print(f"\nClass distribution plot saved as 'class_distribution.png'") 