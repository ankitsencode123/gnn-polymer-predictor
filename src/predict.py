test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)

test_predictions = []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        if batch.x.size(0) == 0:
            continue
        outputs = model(batch)
        test_predictions.append(outputs.cpu())

test_predictions = torch.cat(test_predictions, dim=0).numpy()

print("\nTest predictions generated.")
print(f"Shape of test predictions: {test_predictions.shape}")

final_val_loss = validate_epoch(model, val_loader, criterion, device)
print(f"Final Validation Weighted MAE Loss (using best model): {final_val_loss:.4f}")

test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)

all_predictions = []

print("Making predictions on test set...")
model.eval()
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        if batch.x.size(0) == 0:
            continue
        outputs = model(batch)
        predictions = outputs.cpu().numpy()
        all_predictions.append(predictions)

if len(all_predictions) > 0:
    all_predictions = np.vstack(all_predictions)
    print(f"Predictions shape: {all_predictions.shape}")
    predictions_original_scale = scaler.inverse_transform(all_predictions)
    print(f"Predictions original scale shape: {predictions_original_scale.shape}")
    submission_df = pd.DataFrame({
        'id': [test_df.iloc[i]['id'] for i in test_valid_indices],
        'Tg': predictions_original_scale[:, 0],
        'FFV': predictions_original_scale[:, 1],
        'Tc': predictions_original_scale[:, 2],
        'Density': predictions_original_scale[:, 3],
        'Rg': predictions_original_scale[:, 4]
    })
    missing_ids = set(test_df['id']) - set(submission_df['id'])
    if missing_ids:
        print(f"Missing {len(missing_ids)} predictions for invalid SMILES")
        mean_predictions = np.mean(predictions_original_scale, axis=0)
        for missing_id in missing_ids:
            new_row = {'id': missing_id}
            for i, col in enumerate(property_columns):
                new_row[col] = mean_predictions[i]
            submission_df = pd.concat([submission_df, pd.DataFrame([new_row])], ignore_index=True)
    submission_df = submission_df.sort_values('id').reset_index(drop=True)
    print(f"Final submission shape: {submission_df.shape}")
    print("Submission preview:")
    print(submission_df.head())
    submission_df.to_csv('submission.csv', index=False)
    print("Submission saved as 'submission.csv'")
    print("\nSummary statistics:")
    print(submission_df[property_columns].describe())
else:
    print("No predictions made - check test data processing")
