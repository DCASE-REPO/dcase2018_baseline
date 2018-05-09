#!/usr/bin/env python
"""Class map generator.

Generates a class mapping file that lets us map between class indices and class
names.  Reads standard input and writes to standard output. To generate a usable
class map, pass in the CSV for the entire training set so that we see all
possible labels.

Input: a CSV file with a single header line and matching data lines, e.g.,
  fname,label,manually_verified
  02274ee8.wav,Burping_or_eructation,0

Output: a CSV file with data lines of the form class_index,class_name with one
entry per unique class in the input, and 0-based class indices assigned in
ascending order of class name, e.g.,
  0,Cough
"""

import csv
import sys

unique_labels = sorted(set([row['label'] for row in csv.DictReader(sys.stdin)]))

csv_writer = csv.DictWriter(sys.stdout, fieldnames=['class_index', 'class_name'])
for (class_index, class_name) in enumerate(unique_labels):
  csv_writer.writerow({'class_index': class_index, 'class_name':class_name})
