import pandas as pd

def update_loss_csv(iter_value, loss, filename='loss_per_epoch.csv',type_name="Epoch"):
    # Try to read the existing CSV file
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        # If file does not exist, create a new DataFrame with headers
        df = pd.DataFrame(columns=[type_name, 'Loss'])
    
    # Append the new data to the DataFrame
    new_row = {type_name: iter_value, 'Loss': loss}
    df = df.append(new_row, ignore_index=True)
    
    # Write the updated DataFrame back to the CSV file
    df.to_csv(filename, index=False)