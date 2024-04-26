# get_characters.py

import csv
import sys

def write_characters_to_csv(file_path, characters):
    """ Write the list of characters to a CSV file. """
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        for char in characters:
            writer.writerow([char])

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python get_characters.py <character1> <character2> ...")
        sys.exit(1)
    
    characters = sys.argv[1:]  # Skip the script name
    csv_file_path = 'characters.csv'  # Define the path to the CSV file
    write_characters_to_csv(csv_file_path, characters)
    print(f"Characters saved to {csv_file_path}")
