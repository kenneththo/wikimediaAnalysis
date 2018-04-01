python WikiMediaAnalysis.py -c download -date 20180320
python WikiMediaAnalysis.py -c analysis
python WikiMediaAnalysis.py -c build -n model.h5
python WikiMediaAnalysis.py -c predict -x1 "los angeles" -x2 "san francisco"  -n model.h5
