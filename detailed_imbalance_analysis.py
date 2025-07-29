import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file
df = pd.read_csv('Q2_20230202_majority.csv')

# Count the occurrences of each class
class_counts = df['label_true'].value_counts()

print("DETAILED CLASS IMBALANCE ANALYSIS")
print("=" * 60)
print(f"Dataset: Q2_20230202_majority.csv")
print(f"Total samples: {len(df):,}")
print(f"Number of classes: {len(class_counts)}")
print()

# Basic statistics
print("CLASS DISTRIBUTION:")
print("-" * 30)
for i, (class_name, count) in enumerate(class_counts.items(), 1):
    percentage = (count / len(df)) * 100
    print(f"{i}. {class_name:20} {count:5,} samples ({percentage:5.2f}%)")

print()

# Imbalance metrics
max_count = class_counts.max()
min_count = class_counts.min()
imbalance_ratio = max_count / min_count
gini_coefficient = 1 - sum((class_counts / len(df)) ** 2)

print("IMBALANCE METRICS:")
print("-" * 30)
print(f"Imbalance Ratio (max/min):     {imbalance_ratio:.2f}")
print(f"Gini Coefficient:              {gini_coefficient:.3f}")
print(f"Most frequent class:           {class_counts.index[0]} ({class_counts.iloc[0]:,} samples)")
print(f"Least frequent class:          {class_counts.index[-1]} ({class_counts.iloc[-1]:,} samples)")
print(f"Difference (max - min):        {max_count - min_count:,} samples")

print()

# Severity assessment
print("SEVERITY ASSESSMENT:")
print("-" * 30)
if imbalance_ratio > 3:
    severity = "SEVERE"
    recommendation = "Strongly consider resampling techniques"
elif imbalance_ratio > 2:
    severity = "MODERATE"
    recommendation = "Consider resampling or class weights"
elif imbalance_ratio > 1.5:
    severity = "MILD"
    recommendation = "Monitor performance, consider class weights"
else:
    severity = "MINIMAL"
    recommendation = "Dataset is reasonably balanced"

print(f"Severity Level:                {severity}")
print(f"Recommendation:                {recommendation}")

print()

# Detailed analysis by class
print("DETAILED CLASS ANALYSIS:")
print("-" * 30)
for class_name, count in class_counts.items():
    percentage = (count / len(df)) * 100
    ratio_to_max = count / max_count
    
    if ratio_to_max < 0.5:
        status = "UNDERREPRESENTED"
    elif ratio_to_max > 0.8:
        status = "WELL REPRESENTED"
    else:
        status = "MODERATELY REPRESENTED"
    
    print(f"{class_name:20} {count:5,} samples ({percentage:5.2f}%) - {status}")

print()

# Recommendations
print("RECOMMENDATIONS FOR HANDLING IMBALANCE:")
print("-" * 50)

if imbalance_ratio > 2:
    print("1. DATA RESAMPLING TECHNIQUES:")
    print("   - Oversampling: Duplicate minority class samples")
    print("   - Undersampling: Remove majority class samples")
    print("   - SMOTE: Generate synthetic minority samples")
    print("   - ADASYN: Adaptive synthetic sampling")
    
    print("\n2. ALGORITHM-SPECIFIC TECHNIQUES:")
    print("   - Use class weights in training")
    print("   - Adjust decision thresholds")
    print("   - Use ensemble methods")
    
    print("\n3. EVALUATION METRICS:")
    print("   - Focus on F1-score, precision, recall")
    print("   - Use confusion matrix analysis")
    print("   - Consider ROC-AUC for balanced evaluation")

print("\n4. MONITORING:")
print("   - Track per-class performance metrics")
print("   - Monitor for bias in predictions")
print("   - Validate on balanced test sets")

# Create visualizations
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# 1. Bar plot
sns.barplot(x=class_counts.index, y=class_counts.values, ax=ax1)
ax1.set_title('Class Distribution')
ax1.set_ylabel('Count')
ax1.tick_params(axis='x', rotation=45)

# 2. Pie chart
ax2.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%')
ax2.set_title('Class Distribution (Percentage)')

# 3. Log scale bar plot
sns.barplot(x=class_counts.index, y=class_counts.values, ax=ax3)
ax3.set_yscale('log')
ax3.set_title('Class Distribution (Log Scale)')
ax3.set_ylabel('Count (log scale)')
ax3.tick_params(axis='x', rotation=45)

# 4. Ratio to maximum
ratios = class_counts / max_count
sns.barplot(x=ratios.index, y=ratios.values, ax=ax4)
ax4.set_title('Ratio to Most Frequent Class')
ax4.set_ylabel('Ratio')
ax4.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='50% threshold')
ax4.legend()
ax4.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('detailed_class_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nDetailed analysis plots saved as 'detailed_class_analysis.png'") 