@echo off
py src/validate.py unsimplified test

cd data\results
ren test_srl.json unsimplified_test_srl.json
cd new
ren test_avg_results10doc.csv unsimplified_test_avg_results10doc.csv
cd ..\..\..

py src/validate.py default test

cd data\results
ren test_srl.json simplified_test_srl.json
cd new
ren test_avg_results10doc.csv simplified_test_avg_results10doc.csv
cd ..\..\..

