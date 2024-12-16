# Metadata:
# Required Dependencies: pandas, seaborn, matplotlib, requests
# Usage: python autolysis.py <csv_file>
# This script dynamically installs missing dependencies and performs data analysis.

import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import json
import subprocess
def ensure_dependencies():
    """Ensure all required dependencies are installed."""
    required_packages = ["pandas", "seaborn", "matplotlib", "requests"]
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Installing missing package: {package}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Ensure dependencies are installed before proceeding
ensure_dependencies()

def query_llm(messages, model="gpt-4o-mini"):
    """ 
    Send a request to the AI Proxy for chat completions.

    Args:
        messages (list): A list of message dictionaries for the chat model.
        model (str): The model to use (default is "gpt-4o-mini").

    Returns:
        str: The response content from the model.
    """
    api_key = os.getenv("AIPROXY_TOKEN")
    endpoint = os.getenv("AIPROXY_ENDPOINT", "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions")

    if not api_key:
        raise ValueError("API key not found. Set the AIPROXY_TOKEN environment variable.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    json_data = {
        "model": model,
        "messages": messages
    }

    response = requests.post(endpoint, headers=headers, json=json_data)

    if response.status_code == 200:
        try:
            json_output = response.json()
            return json_output['choices'][0]['message']['content']
        except (KeyError, json.JSONDecodeError):
            raise Exception(f"Unexpected response format: {response.text}")
    else:
        raise Exception(f"Error {response.status_code}: {response.text}")

def data_cleanup_and_preprocessing(dataframe):
    numeric_dataframe = dataframe.select_dtypes(include=["number"])
    numeric_dataframe = numeric_dataframe.fillna(numeric_dataframe.mean())
    numeric_dataframe = (numeric_dataframe - numeric_dataframe.min()) / (numeric_dataframe.max() - numeric_dataframe.min())
    non_numeric_dataframe = dataframe.select_dtypes(exclude=["number"])
    dataframe = pd.concat([numeric_dataframe, non_numeric_dataframe], axis=1)
    return dataframe

def generate_missing_values_bar_chart(dataframe):
    missing_values = dataframe.isnull().sum()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=missing_values.index, y=missing_values.values, palette="viridis")
    plt.xticks(rotation=45, ha='right')
    plt.title("Missing Values per Column")
    plt.ylabel("Number of Missing Values")
    plt.xlabel("Columns")
    plt.savefig("missing_values_bar_chart.png", bbox_inches="tight")
    plt.close()

def generate_generic_visualizations(dataframe):
    numeric_dataframe = dataframe.select_dtypes(include=["number"])
    if not numeric_dataframe.empty:
        plt.figure(figsize=(10, 8))
        sns.heatmap(numeric_dataframe.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title("Correlation Heatmap")
        plt.savefig("correlation_heatmap.png")
        plt.close()
    else:
        print("No numeric columns available for correlation analysis.")

def generate_dynamic_prompts(dataframe):
    num_rows, num_cols = dataframe.shape
    if num_rows > 100000:
        return "Analyze key trends and provide summarized insights due to the dataset's large size."
    elif num_cols > 50:
        return "Focus on high-correlation relationships and missing value patterns given the dataset's complexity."
    else:
        return "Perform a detailed analysis including correlations, outliers, and missing values."

def generate_analysis_insights(dataframe, structured_results):
    prompts = {
        "overview": f"Provide an overview of the dataset with columns: {list(dataframe.columns)}. Include numerical and categorical column insights.",
        "missing_values": f"Narrate a story about the percentage of missing and unique values in each column: {structured_results['missing_values']}",
        "correlation": f"Based on the correlation analysis: {structured_results['correlation']}, identify key insights about strongly correlated columns.",
        "outliers": f"Highlight the columns with the most outliers based on the following analysis: {structured_results['outliers']}"
    }
    insights = {key: query_llm([{"role": "user", "content": prompt}]) for key, prompt in prompts.items()}
    return insights

def generate_readme_part1(insights, structured_results):
    with open("README.md", "w") as readme_file:
        readme_file.write("# Dataset Analysis Report\n\n")
        readme_file.write("## Dataset Overview\n")
        readme_file.write(insights["overview"] + "\n\n")
        readme_file.write("## Missing Values\n")
        readme_file.write(insights["missing_values"] + "\n\n")
        readme_file.write("## Correlation Analysis\n")
        readme_file.write(insights["correlation"] + "\n\n")
        readme_file.write("## Outlier Analysis\n")
        outlier_data = sorted(structured_results['outliers'].items(), key=lambda x: x[1], reverse=True)[:5]
        outlier_info = "\n".join([f"{col}: {count} outliers" for col, count in outlier_data])
        readme_file.write(f"Top 5 columns with the most outliers:\n{outlier_info}\n\n")
        readme_file.write("## Conclusion\n")
        readme_file.write("This report provides a comprehensive overview and insights derived from the dataset.\n")

def generate_readme_part2(insights):
    story_prompt = f"""
    Write a creative, fictional story of about 400 words that ties together the following key insights from the dataset:
    1. The dataset includes these columns: {list(insights['overview'])}.
    2. Missing values are present in the following columns: {insights['missing_values']}.
    3. The most relevant correlations, both positive and negative, are: {insights['correlation']}.
    4. The top 5 columns with the most outliers are: {insights['outliers']}.

    The story should be engaging and informative while including all key insights from the dataset analysis. Ensure it is no longer than 400 words.
    """
    story = query_llm([{"role": "user", "content": story_prompt}])
    with open("README.md", "a") as readme_file:
        readme_file.write("\n## Cool Story Which Ties It Up Together\n")
        readme_file.write(story + "\n")

def main():

    if len(sys.argv) != 2:
        print("Usage: python autolysis.py <csv_file>")
        sys.exit(1)

    csv_file = sys.argv[1]
    if not os.path.exists(csv_file):
        print(f"Error: File '{csv_file}' does not exist.")
        sys.exit(1)

    try:
        dataframe = pd.read_csv(csv_file, encoding='ISO-8859-1')
    except UnicodeDecodeError as e:
        print(f"Error reading CSV file with encoding 'ISO-8859-1': {e}")
        sys.exit(1)

    dataframe = data_cleanup_and_preprocessing(dataframe)

    structured_results = {
        "missing_values": dataframe.isnull().mean().to_dict(),
        "correlation": dataframe.select_dtypes(include=['number']).corr().to_dict(),
        "outliers": {col: ((dataframe[col] < (dataframe[col].mean() - 3 * dataframe[col].std())) | \
                            (dataframe[col] > (dataframe[col].mean() + 3 * dataframe[col].std()))).sum() \
                      for col in dataframe.select_dtypes(include=["number"]).columns}
    }

    generate_missing_values_bar_chart(dataframe)
    generate_generic_visualizations(dataframe)

    insights = generate_analysis_insights(dataframe, structured_results)

    generate_readme_part1(insights, structured_results)
    generate_readme_part2(insights)

    print("Analysis complete. Outputs saved in the current directory.")

if __name__ == "__main__":
    main()

# Metadata
# Dependencies: pandas, seaborn, matplotlib, requests