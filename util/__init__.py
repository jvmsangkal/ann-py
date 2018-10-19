import csv


def write_csv(file_path, rows):
    with open(file_path, 'w') as f:
        csv_writer = csv.writer(f)

        for row in rows:
            csv_writer.writerow(row)
