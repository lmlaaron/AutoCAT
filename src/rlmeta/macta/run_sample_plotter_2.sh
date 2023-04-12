#!/bin/bash

FILES=(
 "trace_500-500-0-FR.txt"
 "trace_502-502-0-FR.txt"
 "trace_505-505-0-FR.txt"
 "trace_548-548-0-FR.txt"
 #"trace_548-548-0-MD.txt"
 #"trace_548-548-0-RR.txt"
 #"trace_548-548-31-FR.txt"
 #"trace_548-548-31-MD.txt"
 #"trace_548-548-31-RR.txt"
 #"trace_548-548-64-FR.txt"
 #"trace_548-548-64-MD.txt"
 #"trace_548-548-64-RR.txt"
 "trace_631-631-0-FR.txt"
 #"trace_631-631-0-MD.txt"
 #"trace_631-631-0-RR.txt"
 #"trace_631-631-31-FR.txt"
 #"trace_631-631-31-MD.txt"
 #"trace_631-631-31-RR.txt"
 #"trace_631-631-64-FR.txt"
 #"trace_631-631-64-MD.txt"
 #"trace_631-631-64-RR.txt"
 "trace_638-638-0-FR.txt"
 #"trace_638-638-0-MD.txt"
 #"trace_638-638-0-RR.txt"
 #"trace_638-638-31-FR.txt"
 #"trace_638-638-31-MD.txt"
 #"trace_638-638-31-RR.txt"
 #"trace_638-638-64-FR.txt"
 #"trace_638-638-64-MD.txt"
 #"trace_638-638-64-RR.txt"
 "trace_641-641-0-FR.txt"
 #"trace_641-641-0-MD.txt"
 #"trace_641-641-0-RR.txt"
 #"trace_641-641-31-FR.txt"
 #"trace_641-641-31-MD.txt"
 #"trace_641-641-31-RR.txt"
 #"trace_641-641-64-FR.txt"
 #"trace_641-641-64-MD.txt"
 #"trace_641-641-64-RR.txt"
 "trace_549-607-0-FR.txt"
 #"trace_549-607-0-MD.txt"
 #"trace_549-607-0-RR.txt"
 #"trace_549-607-0-FR.txt"
 
)

for file in "${FILES[@]}"; do
  echo "Processing file: ${file}"
  python sample_plotter_2.py "${file}"
done