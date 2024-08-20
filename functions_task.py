import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import glob
import numpy as np
import argparse

#Function for preprocessing
def preprocess_data(base_path):
    pr_files = glob.glob(os.path.join(base_path, 'PR', '*', '*.csv'))
    ghi_files = glob.glob(os.path.join(base_path, 'GHI', '*', '*.csv'))
    
    pr_data = pd.concat([pd.read_csv(f) for f in pr_files])
    ghi_data = pd.concat([pd.read_csv(f) for f in ghi_files])
    
    merged_df = pd.merge(pr_data, ghi_data, on='Date')
    merged_df['Date'] = pd.to_datetime(merged_df['Date'])
    merged_df = merged_df.sort_values('Date').reset_index(drop=True)
    
    return merged_df

#Function to generate graph
def create_visualization(df, start_date=None, end_date=None):
    if start_date:
        df = df[df['Date'] >= start_date]
    if end_date:
        df = df[df['Date'] <= end_date]
    
    df['PR_MA'] = df['PR'].rolling(window=30).mean()
    
    start_date = df['Date'].min()
    df['Years'] = (df['Date'] - start_date).dt.total_seconds() / (365.25 * 24 * 60 * 60)
    df['Budget'] = 73.9 * (1 - 0.008) ** df['Years']
    
    plt.figure(figsize=(12, 8))
    
    # Define color ranges for GHI
    color_ranges = [0, 2, 4, 6, np.inf]
    colors = ['navy', 'lightblue', 'orange', 'brown']
    
    for i in range(len(color_ranges) - 1):
        mask = (df['GHI'] >= color_ranges[i]) & (df['GHI'] < color_ranges[i+1])
        plt.scatter(df.loc[mask, 'Date'], df.loc[mask, 'PR'], c=colors[i], s=20)
    
    plt.plot(df['Date'], df['PR_MA'], color='red', linewidth=2)
    plt.plot(df['Date'], df['Budget'], color='darkgreen', linewidth=2)
    
    plt.xlabel('Date')
    plt.ylabel('Performance Ratio [%]')
    plt.title(f'Performance Ratio Evolution\nFrom {df["Date"].min().date()} to {df["Date"].max().date()}')
    
    plt.ylim(0, 100)
    
    plt.text(0.02, 0.02, 'Target Budget Yield Performance Ratio [1Y=73.9%,2Y=73.3%,3Y=72.7%]', 
             transform=plt.gca().transAxes, color='darkgreen')
    plt.text(0.02, 0.06, '30-d moving average of PR', transform=plt.gca().transAxes, color='red')
    
    stats = df.groupby(pd.Grouper(key='Date', freq='D')).mean()
    plt.text(0.75, 0.20, f"Average PR last 7-d: {stats['PR'][-7:].mean():.1f} %", transform=plt.gca().transAxes)
    plt.text(0.75, 0.17, f"Average PR last 30-d: {stats['PR'][-30:].mean():.1f} %", transform=plt.gca().transAxes)
    plt.text(0.75, 0.14, f"Average PR last 60-d: {stats['PR'][-60:].mean():.1f} %", transform=plt.gca().transAxes)
    plt.text(0.75, 0.11, f"Average PR last 90-d: {stats['PR'][-90:].mean():.1f} %", transform=plt.gca().transAxes)
    plt.text(0.75, 0.08, f"Average PR last 365-d: {stats['PR'][-365:].mean():.1f} %", transform=plt.gca().transAxes)
    plt.text(0.75, 0.05, f"Average PR Lifetime: {stats['PR'].mean():.1f} %", transform=plt.gca().transAxes, fontweight='bold')
    
    above_budget = (df['PR'] > df['Budget']).sum()
    total_points = len(df)
    percentage_above = (above_budget / total_points) * 100
    plt.text(0.02, 0.10, f"Points above Target Budget PR = {above_budget}/{total_points} = {percentage_above:.1f}%", 
             transform=plt.gca().transAxes)
    
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='< 2',
                   markerfacecolor='navy', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='2 - 4',
                   markerfacecolor='lightblue', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='4 - 6',
                   markerfacecolor='orange', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='> 6',
                   markerfacecolor='brown', markersize=10)
    ]
    
    legend = plt.legend(handles=legend_elements, loc='upper left', title='Daily Irradiation [kWh/m2]',
                        bbox_to_anchor=(0.01, 0.99), frameon=False, ncol=4)
    plt.setp(legend.get_title(), fontsize='small')
    
    plt.tight_layout()
    plt.show()

#Comment out below portion to run without arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate PR graph with optional date range.')
    parser.add_argument('--start_date', type=str, help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end_date', type=str, help='End date in YYYY-MM-DD format')
    args = parser.parse_args()

    base_path = '.'  # Assuming the script is run in the same directory as PR and GHI folders
    df = preprocess_data(base_path)
    
    start_date = pd.to_datetime(args.start_date) if args.start_date else None
    end_date = pd.to_datetime(args.end_date) if args.end_date else None
    
    create_visualization(df, start_date, end_date)