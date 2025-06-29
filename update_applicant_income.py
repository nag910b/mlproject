import pandas as pd

# Read the CSV file
df = pd.read_csv("notebook/data/LoanApprovalPrediction.csv")

# Display current applicant income statistics
print("Current Applicant Income Statistics:")
print(f"Mean: {df['ApplicantIncome'].mean():.2f}")
print(f"Min: {df['ApplicantIncome'].min()}")
print(f"Max: {df['ApplicantIncome'].max()}")
print(f"First 5 values: {df['ApplicantIncome'].head().tolist()}")

# Add 50000 to all applicant income values
df['ApplicantIncome'] = df['ApplicantIncome'] + 50000

# Display updated applicant income statistics
print("\nUpdated Applicant Income Statistics:")
print(f"Mean: {df['ApplicantIncome'].mean():.2f}")
print(f"Min: {df['ApplicantIncome'].min()}")
print(f"Max: {df['ApplicantIncome'].max()}")
print(f"First 5 values: {df['ApplicantIncome'].head().tolist()}")

# Save the updated data back to the CSV file
df.to_csv("notebook/data/LoanApprovalPrediction.csv", index=False)

print("\n✅ Successfully added 50000 to all applicant income values!")
print("The CSV file has been updated.") 