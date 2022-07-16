for ((i=1; i <= $1; i++))
do
  echo $i
  python ./andrewnet.py TRAIN
  python ./andrewnet.py TEST_A
done