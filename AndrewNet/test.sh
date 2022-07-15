for run in {1..$1}; do
  echo "Run $run"
  python ./andrewnet.py TRAIN
  python ./andrewnet.py TEST_A
done